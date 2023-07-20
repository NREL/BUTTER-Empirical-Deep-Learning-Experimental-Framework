from __future__ import annotations

from itertools import chain
from typing import Any, Iterable, Sequence, Set, Type, List
from numbers import Number

import numpy
import pandas
import pandas.core.groupby.groupby

from dmp.task.experiment.experiment_summary_record import ExperimentSummaryRecord
from dmp.task.experiment.training_experiment.training_experiment_keys import (
    TrainingExperimentKeys,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dmp.task.experiment.training_experiment.training_experiment import (
        TrainingExperiment,
    )

for k, v in {
    "display.max_rows": 9000,
    "display.min_rows": 40,
    "display.max_columns": None,
    "display.width": 240,
}.items():
    pandas.set_option(k, v)


class TrainingExperimentSummarizer:
    def summarize(
        self,
        experiment: TrainingExperiment,
        results: List[pandas.DataFrame],
    ) -> ExperimentSummaryRecord:
        keys: TrainingExperimentKeys = experiment.keys
        sources = []
        for i, run_history in enumerate(results):
            run_history[keys.run] = i
            sources.append(run_history)
        num_sources = len(sources)
        history = pandas.concat(sources, ignore_index=True, axis=0)
        runs = numpy.arange(num_sources)
        del sources
        del results

        # remove duplicate loss columns
        for metric in keys.loss_metrics:
            for prefix in keys.data_sets:
                column = prefix + "_" + metric
                loss_column = prefix + "_" + keys.loss
                if column in history and loss_column in history:
                    if history[column].equals(history[loss_column]):
                        del history[column]

        history.set_index([keys.run, keys.epoch], inplace=True)
        history.sort_index(inplace=True)

        for (
            column,
            cfunc,
            ifunc,
            result_column,
            epoch_column,
        ) in keys.run_summary_metrics:
            if column in history:
                history[result_column] = numpy.nan
                history[epoch_column] = 0
                history[result_column] = history[result_column].astype(numpy.float32)
                history[epoch_column] = history[epoch_column].astype(numpy.int16)
                for run in runs:
                    run_history = history.loc[(run, slice(None)), column]
                    cumulative_values, cumulative_indexes = cfunc(
                        run_history.to_numpy()
                    )
                    history.loc[(run, slice(None)), result_column] = cumulative_values
                    cumulative_epochs = run_history.iloc[
                        cumulative_indexes
                    ].index.get_level_values(keys.epoch)
                    history.loc[(run, slice(None)), epoch_column] = cumulative_epochs

        selected_epochs = self._select_epochs(
            history.index.get_level_values(keys.epoch)
        )

        history[keys.canonical_epoch] = False
        history[keys.canonical_epoch] = history[keys.canonical_epoch].astype(
            numpy.bool_
        )
        history.loc[(slice(None), selected_epochs), keys.canonical_epoch] = True

        epoch_subset = self._summarize_epoch_subset(
            experiment, history, selected_epochs
        )
        # print(epoch_subset.describe())
        # print(epoch_subset)

        # remove epoch start times from summary
        if keys.epoch_start_time_ms in history:
            del history[keys.epoch_start_time_ms]
        by_epoch = self._summarize_by_epoch(experiment, history, selected_epochs)

        # print(by_epoch.head(200))
        # print(by_epoch.describe())

        by_loss = self._summarize_by_loss(experiment, runs, history)

        return ExperimentSummaryRecord(
            num_sources,
            by_epoch,
            by_loss,
            None,
            epoch_subset,
        )

    def _select_epochs(
        self,
        epochs: pandas.Index,
    ) -> numpy.ndarray:
        return numpy.unique(
            numpy.round(
                self.make_summary_points(
                    epochs.min(),
                    epochs.max(),
                    128,
                    1,
                    numpy.log(10.0 / 1) / 100,
                )
            ).astype(numpy.int16)
        )

    def _summarize_by_epoch(
        self,
        experiment: TrainingExperiment,
        history: pandas.DataFrame,
        selected_epochs: numpy.ndarray,
    ) -> pandas.DataFrame:
        keys: TrainingExperimentKeys = experiment.keys
        epoch_samples = history.loc[(slice(None), selected_epochs), :]

        skip_set = {keys.run, keys.epoch}
        by_epoch = self._summarize_group(
            experiment,
            epoch_samples.groupby(keys.epoch, sort=True),
            keys.epoch,
            {k for k in keys.simple_summarize_keys if k not in skip_set},
            history.columns,
        )

        return by_epoch

    def _summarize_epoch_subset(
        self,
        experiment: TrainingExperiment,
        history: pandas.DataFrame,
        selected_epochs: numpy.ndarray,
    ) -> pandas.DataFrame:
        keys: TrainingExperimentKeys = experiment.keys

        run_groups = history.groupby(keys.run)
        selection = [history.loc[(slice(None), selected_epochs), :].index.values]
        for (
            column,
            cfunc,
            ifunc,
            result_column,
            epoch_column,
        ) in keys.run_summary_metrics:
            if column in history:
                selection.append(ifunc(run_groups[column]))
        selected_rows = history.loc[numpy.unique(numpy.concatenate(selection))]
        del selection

        selected_rows.sort_index(inplace=True)
        # print(selected_rows[selected_rows[keys.canonical_epoch] == False])
        return selected_rows

    def _summarize_by_loss(
        self,
        experiment: TrainingExperiment,
        runs: numpy.ndarray,
        history: pandas.DataFrame,
    ) -> pandas.DataFrame:
        keys: TrainingExperimentKeys = experiment.keys
        min_median = history[keys.test_loss_cmin].groupby(keys.run).min().median()
        loss_levels = numpy.flip(
            self.make_summary_points(
                min_median,
                history[keys.test_loss_cmin].groupby(keys.run).max().median(),
                1e-5,
                1e-4,
                numpy.log(1.0 / 0.1) / 10,
            ).astype(numpy.float32)
        )

        # print(f'min {min_pt} max {max_pt}')

        # print(loss_levels)
        # print(loss_levels.shape)

        loss_series = history[keys.test_loss_cmin]
        interpolated_loss_points = {
            k: [] for k in chain((keys.run, keys.epoch), history.columns)
        }
        for run in runs:
            run_df = history.loc[run, :]
            losses = run_df[keys.test_loss_cmin]
            loss_level_idx = 0
            prev_epoch: Any = None
            for curr_epoch, curr_loss in losses.items():
                while curr_loss <= loss_levels[loss_level_idx]:
                    loss_level = loss_levels[loss_level_idx]
                    if prev_epoch is None:
                        prev_epoch = curr_epoch
                    prev_loss = losses.loc[prev_epoch]

                    delta = prev_loss - curr_loss
                    curr_weight = 0.5
                    if delta > 1e-18:
                        curr_weight = (loss_level - curr_loss) / delta
                    prev_weight = 1 - curr_weight

                    prev_index = (run, prev_epoch)
                    prev = history.loc[prev_index]

                    curr_index = (run, curr_epoch)
                    curr = history.loc[curr_index]  # type: ignore

                    interpolated_loss_points[keys.run].append(run)
                    interpolated_loss_points[keys.epoch].append(
                        curr_weight * curr_epoch + prev_weight * prev_epoch
                    )

                    for c in history.columns:
                        prev_value = prev[c]
                        curr_value = curr[c]  # type:ignore
                        interpolated_value = None
                        if isinstance(curr_value, Number):
                            if pandas.isna(prev_value) or pandas.isna(
                                curr_value
                            ):  # type: ignore
                                interpolated_value = curr_value
                            else:
                                interpolated_value = (
                                    curr_weight * curr_value + prev_weight * prev_value
                                )
                        else:
                            if curr_weight >= prev_weight:
                                interpolated_value = curr_value
                            else:
                                interpolated_value = prev_value

                        interpolated_loss_points[c].append(interpolated_value)

                    interpolated_loss_points[keys.test_loss_cmin][-1] = loss_level

                    loss_level_idx += 1
                    if loss_level_idx >= loss_levels.size:
                        break

                if loss_level_idx >= loss_levels.size:
                    break
                prev_epoch = curr_epoch

        # print(interpolated_loss_points)

        skip_set = {keys.run, keys.test_loss_cmin}
        by_loss = self._summarize_group(
            experiment,
            pandas.DataFrame(interpolated_loss_points).groupby(
                keys.test_loss_cmin, sort=True
            ),
            keys.test_loss_cmin,
            {k for k in keys.simple_summarize_keys if k not in skip_set},
            history.columns,
        )

        # print(by_loss)
        # print(by_loss.describe())
        return by_loss

    def make_summary_points(
        self,
        min_pt: float,
        max_pt: float,
        switch_point: float,
        linear_resolution: float,
        logarithmic_resolution: float,
    ) -> numpy.ndarray:
        linear_points = numpy.arange(
            min_pt, min(switch_point, max_pt) + linear_resolution, linear_resolution
        )
        logarithmic_points = numpy.exp(
            numpy.arange(
                numpy.log(max(min_pt, switch_point) + linear_resolution),
                numpy.log(max_pt + logarithmic_resolution),
                logarithmic_resolution,
            )
        )

        return numpy.concatenate((linear_points, logarithmic_points))

    def _summarize_group(
        self,
        experiment: TrainingExperiment,
        groups: pandas.core.groupby.groupby.GroupBy,
        group_column: str,  #: Union[numpy.ndarray, pandas.Series],
        simple_metrics: Set[str],
        quantile_metrics: Iterable,
    ) -> pandas.DataFrame:
        keys: TrainingExperimentKeys = experiment.keys
        by_loss = pandas.DataFrame(
            {
                group_column: [group for group, _ in groups],
                keys.count: groups.size(),
            },
        )
        by_loss.set_index(group_column, inplace=True)

        for key in simple_metrics:
            if key not in key in ignore_metrics and key in groups:  # type: ignore
                by_loss[key + "_quantile_50"] = (
                    groups[key].median().astype(numpy.float32)
                )

        quantile_points = numpy.array([0, 0.25, 0.5, 0.75, 1], dtype=numpy.float32)
        quantile_metrics = [
            k for k in quantile_metrics if k not in by_loss and k not in simple_metrics
        ]
        quantiles = (
            groups[quantile_metrics]
            .quantile(quantile_points)  # type: ignore
            .unstack()
            .astype(numpy.float32)
        )
        quantiles.columns = [
            f"{metric}_quantile_{int(quantile * 100)}"
            for metric, quantile in quantiles.columns.to_flat_index().values
        ]

        for key in chain(simple_metrics, quantile_metrics):
            if key in groups:
                by_loss[key + "_count"] = groups[key].count().astype(numpy.int16)

        by_loss = pandas.concat(
            (
                by_loss,
                quantiles,
            ),
            axis=1,
        )

        return by_loss


summarizer = TrainingExperimentSummarizer()
