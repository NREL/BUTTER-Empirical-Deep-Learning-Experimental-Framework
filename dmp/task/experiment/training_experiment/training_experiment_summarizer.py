from itertools import chain
from typing import Iterable, Sequence, Set, Type
import numpy
import pandas
import pandas.core.groupby.groupby

from dmp.task.experiment.experiment_result_record import ExperimentResultRecord
from dmp.task.experiment.experiment_summary_record import ExperimentSummaryRecord
from dmp.task.experiment.training_experiment.training_experiment_keys import TrainingExperimentKeys


class TrainingExperimentSummarizer():
    
    def summarize(
        self,
        cls: Type['TrainingExperiment'],
        results: Sequence[ExperimentResultRecord],
    ) -> ExperimentSummaryRecord:
        keys: TrainingExperimentKeys = cls.keys
        sources = []
        for i, r in enumerate(results):
            run_history = r.run_history
            run_history[keys.run] = i

            for metric in keys.loss_metrics:
                if metric in run_history:
                    run_history[metric + '_cmin'] = run_history[metric].cummin()

            for metric in ('test_accuracy', 'validation_accuracy'):
                if metric in run_history:
                    run_history[metric + '_cmax'] = run_history[metric].cummax()

            sources.append(run_history)
        history = pandas.concat(sources, ignore_index=True, axis=0)

        if keys.epoch_start_time_ms in history:
            del history[keys.epoch_start_time_ms]

        history.set_index([keys.run, keys.epoch], inplace=True)
        history.sort_index(inplace=True)
        by_epoch = cls._summarize_by_epoch(history)

        # print(by_epoch.head(100))
        print(by_epoch.describe())

        by_loss = cls._summarize_by_loss(history)

        return ExperimentSummaryRecord(
            by_epoch,
            by_loss,
            None,
            None,
        )

    
    def _summarize_by_loss(self,
        cls: Type['TrainingExperiment'],
        history: pandas.DataFrame,
        ) -> pandas.DataFrame:
        keys = cls.keys
        loss_levels = numpy.flip(
            cls.make_summary_points(
                history[keys.test_loss_cmin].groupby(keys.run).min().median(),
                history.loc[(slice(None),
                             slice(0, 1)), :][keys.test_loss_cmin].median(),
                1e-9,
                1e-11,
                numpy.log(1.0 / .1) / 500,
            ).astype(numpy.float32))

        # print(f'min {min_pt} max {max_pt}')

        print(loss_levels)
        print(loss_levels.shape)

        loss_series = history[keys.test_loss_cmin]
        interpolated_loss_points = {
            k: []
            for k in itertools.chain((keys.run, keys.epoch), history.columns)
        }
        for run in loss_series.index.unique(keys.run):
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

                    curr_weight = (loss_level - curr_loss) / (prev_loss -
                                                              curr_loss)
                    prev_weight = 1 - curr_weight

                    prev_index = (run, prev_epoch)
                    prev = history.loc[prev_index]

                    curr_index = (run, curr_epoch)
                    curr = history.loc[curr_index]  # type: ignore

                    interpolated_loss_points[keys.run].append(run)
                    interpolated_loss_points[keys.epoch].append(
                        curr_weight * curr_epoch + prev_weight * prev_epoch)

                    for c in history.columns:
                        prev_value = prev[c]
                        curr_value = curr[c]
                        interpolated_value = None
                        if isinstance(curr_value, Number):
                            if pandas.isna(prev_value) or \
                                pandas.isna(curr_value): # type: ignore
                                interpolated_value = curr_value
                            else:
                                interpolated_value = curr_weight * curr_value + prev_weight * prev_value
                        else:
                            if curr_weight >= prev_weight:
                                interpolated_value = curr_value
                            else:
                                interpolated_value = prev_value

                        interpolated_loss_points[c].append(interpolated_value)

                    interpolated_loss_points[
                        keys.test_loss_cmin][-1] = loss_level

                    loss_level_idx += 1
                    if loss_level_idx >= loss_levels.size:
                        break

                if loss_level_idx >= loss_levels.size:
                    break
                prev_epoch = curr_epoch

        # print(interpolated_loss_points)

        skip_set = {keys.run, keys.test_loss_cmin}
        by_loss = cls._summarize_group(
            pandas.DataFrame(interpolated_loss_points).groupby(
                keys.test_loss_cmin, sort=True),
            keys.test_loss_cmin,
            {k
             for k in keys.simple_summarize_keys if k not in skip_set},
            history.columns,
        )

        # print(by_loss)
        print(by_loss.describe())
        return by_loss

    
    def _summarize_by_epoch(self, cls: Type['TrainingExperiment'],
                            history: pandas.DataFrame,) -> pandas.DataFrame:
        keys = cls.keys
        epochs = history.index.get_level_values(keys.epoch)

        epoch_selections = numpy.unique(
            numpy.round(
                cls.make_summary_points(
                    epochs.min(),
                    epochs.max(),
                    128,
                    1,
                    numpy.log(10.0 / 1) / 100,
                )).astype(numpy.int16))

        epoch_selection = numpy.concatenate(epoch_selections)
        print(epoch_selection)
        print(epoch_selection.size)

        epoch_samples = history.loc[(slice(None), epoch_selection), :]

        skip_set = {keys.run, keys.epoch}
        by_epoch = cls._summarize_group(
            epoch_samples.groupby(keys.epoch, sort=True),
            keys.epoch,
            {k
             for k in keys.simple_summarize_keys if k not in skip_set},
            history.columns,
        )

        return by_epoch

    def make_summary_points(
        self,
        cls: Type['TrainingExperiment'],
        min_pt: float,
        max_pt: float,
        switch_point: float,
        linear_resolution: float,
        logarithmic_resolution: float,
    ) -> numpy.ndarray:
        linear_points = numpy.arange(
            min_pt,
            min(switch_point, max_pt) + linear_resolution, linear_resolution)
        logarithmic_points = numpy.exp(
            numpy.arange(
                numpy.log(max(min_pt, switch_point) + linear_resolution),
                numpy.log(max_pt + logarithmic_resolution),
                logarithmic_resolution,
            ))

        return numpy.concatenate((linear_points, logarithmic_points))

    def _summarize_group(
        self,
        cls: Type['TrainingExperiment'],
        groups: pandas.core.groupby.groupby.GroupBy,
        group_column: str,  #: Union[numpy.ndarray, pandas.Series],
        simple_metrics: Set[str],
        quantile_metrics: Iterable,
    ) -> pandas.DataFrame:
        keys = cls.keys
        by_loss = pandas.DataFrame(
            {
                group_column: [group for group, _ in groups],
                keys.count: groups.size(),
            }, )
        by_loss.set_index(group_column, inplace=True)

        for key in simple_metrics:
            if key not in key in ignore_metrics and key in groups:  # type: ignore
                by_loss[key + '_quantile_50'] = groups[key].median().astype(
                    numpy.float32)

        quantile_points = numpy.array([0, .25, .5, .75, 1],
                                      dtype=numpy.float32)
        quantile_metrics = [
            k for k in quantile_metrics
            if k not in by_loss and k not in simple_metrics
        ]
        quantiles = groups[quantile_metrics].quantile(  # type: ignore
            quantile_points).unstack().astype(numpy.float32)
        quantiles.columns = [
            f'{metric}_quantile_{int(quantile * 100)}'
            for metric, quantile in quantiles.columns.to_flat_index().values
        ]

        for key in chain(simple_metrics, quantile_metrics):
            if key in groups:
                by_loss[key + '_count'] = groups[key].count().astype(
                    numpy.int16)

        by_loss = pandas.concat(
            (
                by_loss,
                quantiles,
            ),
            axis=1,
        )

        return by_loss

summarizer = TrainingExperimentSummarizer()

from dmp.task.experiment.training_experiment.training_experiment import TrainingExperiment