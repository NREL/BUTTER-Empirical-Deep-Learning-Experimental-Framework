from __future__ import annotations

from itertools import chain
from typing import Any, Iterable, Optional, Sequence, Set, Tuple, Type, List
from numbers import Number
from uuid import UUID

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


def is_valid_number(v):
    return (
        isinstance(v, Number)
        and not pandas.isna(v)  # type: ignore
        and not numpy.isfinite(v)  # type: ignore
        and not numpy.isnan(v)  # type: ignore
    )


class TrainingExperimentSummarizer:
    def summarize(
        self,
        experiment: TrainingExperiment,
        histories: List[Tuple[UUID, pandas.DataFrame]],
    ) -> Optional[ExperimentSummaryRecord]:
        keys: TrainingExperimentKeys = experiment.keys
        sources = []
        for i, (run_id, run_history) in enumerate(histories):
            print(f"run {i} {run_id} {run_history.shape}")
            if "index" in run_history.columns:
                del run_history["index"]

            run_history[keys.run] = i
            sources.append(run_history)
        num_sources = len(sources)
        if num_sources <= 0:
            print(f"No Sources to summarize.")
            return

        history = pandas.concat(sources, ignore_index=True, axis=0)
        runs = numpy.arange(num_sources)
        del sources
        del histories

        # remove duplicate loss columns
        for metric in keys.loss_metrics:
            for prefix in keys.data_sets:
                column = prefix + "_" + metric
                loss_column = prefix + "_" + keys.loss
                if column in history and loss_column in history:
                    if history[column].equals(history[loss_column]):
                        del history[column]

        history.set_index([keys.run, keys.epoch], inplace=True, drop=False)
        history.sort_index(inplace=True)
        print(f"history 3:")
        print(history.head(10))

        for (
            column,
            cfunc,
            ifunc,
            result_column,
            epoch_column,
        ) in keys.run_summary_metrics:
            if column in history:
                history[result_column] = numpy.nan
                history[result_column] = history[result_column].astype(numpy.float32)
                history[epoch_column] = 0
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

        # print(history)
        selected_epochs = self._select_epochs(
            history.index.get_level_values(keys.epoch)
        )
        # print(f"selected epochs: {selected_epochs}")

        # history[keys.canonical_epoch] = False
        # history[keys.canonical_epoch] = history[keys.canonical_epoch].astype(
        #     numpy.bool_
        # )
        # history.loc[(slice(None), selected_epochs), keys.canonical_epoch] = True

        # epoch_subset = self._summarize_epoch_subset(
        #     experiment, history, selected_epochs
        # )
        # print(epoch_subset.describe())
        # print(epoch_subset)

        # remove epoch start times from summary
        if keys.epoch_start_time_ms in history:
            del history[keys.epoch_start_time_ms]
        by_epoch = self._summarize_by_epoch(experiment, history, selected_epochs)

        for k, v in {
            "display.max_rows": 9000,
            "display.min_rows": 40,
            "display.max_columns": None,
            "display.width": 300,
        }.items():
            pandas.set_option(k, v)

        print(by_epoch.head(3000))
        print(by_epoch.describe())

        by_loss = self._summarize_by_loss(experiment, runs, history)
        print(by_loss.head(3000))
        print(by_loss.describe())

        return ExperimentSummaryRecord(
            num_sources,
            by_epoch,
            by_loss,
            None,
            None,
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
                    1,
                    32,
                    numpy.log(10 / 1) / 50,
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
        del epoch_samples[keys.run]
        del epoch_samples[keys.epoch]

        print(f"summarize by epoch names {history.columns.values.tolist()}")
        print(history.columns)
        print(history.head(10))

        skip_set = {keys.run, keys.epoch}
        by_epoch = self._summarize_group(
            experiment,
            epoch_samples.groupby(keys.epoch, sort=True),
            keys.epoch,
            {k for k in keys.simple_summarize_keys if k not in skip_set},
            history.columns.values.tolist(),
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
        loss_index_key = "_loss_index"
        keys: TrainingExperimentKeys = experiment.keys
        loss_key = keys.test_loss_cmin
        history = history.copy()
        del history[keys.run]

        run_groups = history[loss_key].groupby(keys.run)
        loss_levels = numpy.flip(
            self.make_summary_points(
                run_groups.min().median(),
                run_groups.max().min(),
                1e-8,
                1e-6,
                numpy.log(10) / 25,
            ).astype(numpy.float32)
        )

        # print(f"loss_levels")
        # print(loss_levels)
        # print(loss_levels.shape)

        # loss_series = history[keys.test_loss_cmin]
        interpolated_loss_points = {
            k: [] for k in chain((loss_index_key, keys.epoch), history.columns)
        }
        for run in runs:
            run_df = history.loc[run, :].set_index(keys.epoch, drop=False).sort_index()
            # del run_df[keys.run]
            # run_df.sort_index()

            # loss_level_idx = 0
            # loss_level = loss_levels[loss_level_idx]

            # for i in range(len(run_df)):
            i = 0
            for loss_index, loss_level in enumerate(loss_levels):
                while i < len(run_df) and run_df[loss_key].iloc[i] > loss_level:
                    i += 1

                if i >= len(run_df):
                    break

                curr = run_df.iloc[i]
                curr_loss = curr[loss_key]

                prev = run_df.iloc[max(0, i - 1)]
                prev_loss = prev[loss_key]

                prev_weight = 0.0
                delta = prev_loss - curr_loss
                if delta > 1e-12:
                    prev_weight = (loss_level - curr_loss) / delta
                curr_weight = 1.0 - prev_weight

                # interpolated_loss_points[keys.run].append(run)
                # interpolated_loss_points[keys.epoch].append(
                #     curr_weight * curr[keys.epoch] + prev_weight * prev[keys.epoch]
                # )
                interpolated_loss_points[loss_index_key].append(loss_index)
                # print(
                #     f"intrp: {loss_index} {i} {prev_loss} {loss_level} {curr_loss} -> {prev_weight}"
                # )
                for c in run_df.columns:
                    prev_value = prev[c]
                    curr_value = curr[c]
                    interpolated_value = curr_value
                    if (
                        is_valid_number(curr_value)
                        and is_valid_number(prev_value)
                        and is_valid_number(curr_weight)
                        and is_valid_number(prev_weight)
                    ):
                        interpolated_value = (
                            curr_weight * curr_value + prev_weight * prev_value  # type: ignore
                        )
                        # print(
                        #     f"intrpc: {c} {prev_value} {curr_value} -> {interpolated_value}"
                        # )

                    interpolated_loss_points[c].append(interpolated_value)

        # print("interpolated_loss_points")
        # print(interpolated_loss_points)

        by_loss_df = pandas.DataFrame(interpolated_loss_points)
        by_loss_df.set_index(loss_index_key, inplace=True, drop=True)
        by_loss_df.sort_index(inplace=True)

        skip_set = {keys.run, keys.test_loss_cmin}
        by_loss = self._summarize_group(
            experiment,
            by_loss_df.groupby(loss_key, sort=True),
            keys.test_loss_cmin,
            {k for k in keys.simple_summarize_keys if k not in skip_set},
            history.columns.values.tolist(),
        )

        # print(by_loss)
        # print(by_loss.describe())
        return by_loss

    def make_summary_points(
        self,
        min_pt: float,
        max_pt: float,
        linear_resolution: float,
        switch_point: float,
        logarithmic_resolution: float,
    ) -> numpy.ndarray:
        # linear_points = numpy.arange(
        #     min_pt, min(switch_point, max_pt) + linear_resolution, linear_resolution
        # )

        # print(
        #     f"make_summary_points {min_pt} {max_pt} {linear_resolution} {switch_point} {logarithmic_resolution}"
        # )
        switch_point = numpy.floor(switch_point / linear_resolution) * linear_resolution

        linear_start = numpy.ceil(min_pt / linear_resolution)
        linear_max = min(switch_point, max_pt)
        linear_end = numpy.floor(linear_max / linear_resolution)

        # print(f"start {linear_start}, end {linear_end}")
        linear_points = (
            numpy.fromiter(
                range(int(linear_start), int(linear_end) + 1),
                numpy.dtype(numpy.float32),
            )
            * linear_resolution
        )

        # print(f"linear_points {linear_points}")

        if min_pt > switch_point:
            log_start = numpy.ceil(numpy.log(min_pt) / logarithmic_resolution)
        else:
            switch_point_log_index = numpy.log(switch_point) / logarithmic_resolution
            log_start = numpy.ceil(numpy.log(switch_point) / logarithmic_resolution)
            if switch_point_log_index >= log_start:
                log_start += 1

        log_end = numpy.floor(numpy.log(max_pt) / logarithmic_resolution)
        # print(f"log_start {log_start}, log_end {log_end}")

        log_points = numpy.exp(
            numpy.fromiter(
                range(
                    int(log_start),
                    int(log_end) + 1,
                ),
                numpy.dtype(numpy.float32),
            )
            * logarithmic_resolution
        )

        # print(f"log_points {log_points}")

        return numpy.concatenate((linear_points, log_points))

        # return numpy.concatenate((linear_points, logarithmic_points))

    def _summarize_group(
        self,
        experiment: TrainingExperiment,
        groups: pandas.core.groupby.groupby.GroupBy,
        group_column: str,  #: Union[numpy.ndarray, pandas.Series],
        simple_metrics: Set[str],
        quantile_metrics: Iterable,
    ) -> pandas.DataFrame:
        keys: TrainingExperimentKeys = experiment.keys
        result = pandas.DataFrame(
            {
                group_column: [group for group, _ in groups],
                # keys.count: groups.size(),
            },
        )
        result.set_index(group_column, inplace=True)

        print(f"summarize group {len(groups)} {group_column} {simple_metrics}")
        print(result)
        for metric in simple_metrics:
            if metric in groups.obj:  # type: ignore
                result[metric + "_quantile_50"] = (
                    groups[metric].median().astype(numpy.float32)
                )
        print(f"summarize group 2")
        print(result)

        quantile_points = numpy.array([0, 0.25, 0.5, 0.75, 1], dtype=numpy.float32)
        from pandas.api.types import is_numeric_dtype

        print(f"summarize group 3 input quantile metrics: {quantile_metrics}")
        print(f"in result: {[key for key in quantile_metrics if key in result]}")
        print(
            f"in simple: {[key for key in quantile_metrics if key in simple_metrics]}"
        )
        print(f"in groups: {[key for key in quantile_metrics if key in groups.obj]}")
        print(
            f"numeric: {[key for key in quantile_metrics if key in groups.obj and is_numeric_dtype(groups.obj[key])]}"
        )
        quantile_metrics = [
            key
            for key in quantile_metrics
            if (
                key not in result
                and key not in simple_metrics
                and key in groups.obj
                and is_numeric_dtype(groups.obj[key])
            )
        ]

        print(f"summarize group 3 quantiles: {quantile_metrics}")

        quantiles = (
            groups[quantile_metrics]
            .quantile(quantile_points, interpolation="linear")  # type: ignore
            .unstack()
            .astype(numpy.float32)
        )
        quantiles.columns = [
            f"{metric}_quantile_{int(quantile * 100)}"
            for metric, quantile in quantiles.columns.to_flat_index().values
        ]

        for key in chain(simple_metrics, quantile_metrics):
            if key in groups.obj:
                result[key + "_count"] = (
                    pandas.notna(groups[key]).count().astype(numpy.int16)
                )

        result = pandas.concat(
            (
                result,
                quantiles,
            ),
            axis=1,
        )

        # print(f"summarize group 4")
        # print(result)

        return result


summarizer = TrainingExperimentSummarizer()
