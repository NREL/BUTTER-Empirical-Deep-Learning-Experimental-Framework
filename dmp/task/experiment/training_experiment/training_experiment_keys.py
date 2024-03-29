from itertools import chain
from typing import Callable, Iterable, List, Sequence, Set, Tuple

import numpy


class TrainingExperimentKeys:
    def __init__(self) -> None:
        self.run: str = "run"
        self.epoch: str = "epoch"
        self.count: str = "count"

        self.test_loss_cmin: str = "test_loss_cumulative_min"
        self.canonical_epoch: str = "canonical_epoch"

        self.train: str = "train"
        self.test: str = "test"
        self.validation: str = "validation"

        self.trained: str = "trained"

        self.test_data_sets: Sequence[str] = (
            self.test,
            self.validation,
        )

        self.data_sets: Sequence[str] = (self.train,) + self.test_data_sets
        self.measurement_prefixes = (self.trained, *self.data_sets)

        self.loss = "loss"
        self.accuracy = "accuracy"

        self.best = "best"
        self.cumulative_best = "best"
        self.cepoch = "best_epoch"

        self.test_loss = self.test + "_" + self.loss
        self.test_loss_cmin = self.test_loss + "_" + self.cumulative_best

        self.validation_loss = self.validation + "_" + self.loss
        self.validation_loss_cmin = self.validation_loss + "_" + self.cumulative_best

        self.loss_metrics = [
            self.loss,
            self.accuracy,
            "mean_squared_error",
            "binary_crossentropy",
            "catergorical_crossentropy",
        ]

        def make_with_prefixes(
            prefixes: Iterable[str],
            keys: Iterable[str],
        ) -> List[str]:
            return list(chain(*[[p + "_" + k for k in keys] for p in prefixes]))

        def make_with_data_set_prefixes(keys: Iterable[str]) -> List[str]:
            return make_with_prefixes(self.measurement_prefixes, keys)

        self.train_start_timestamp: str = "train_start_timestamp"

        self.interval_suffix: str = "ms"
        self.epoch_start_time_ms: str = "epoch_start" + self.interval_suffix
        self.epoch_time_ms: str = "train" + self.interval_suffix

        self.extended_history_columns: Set[str] = set(
            make_with_data_set_prefixes(
                (
                    "cosine_similarity",
                    "kullback_leibler_divergence",
                    "root_mean_squared_error",
                    "mean_absolute_error",
                    "mean_squared_logarithmic_error",
                    "hinge",
                    "squared_hinge",
                    "categorical_hinge",
                )
            )
        )

        # self.loss_metrics: Sequence[str] = (
        #     "categorical_crossentropy",
        #     "mean_squared_error",
        #     "binary_crossentropy",
        # )

        self.prefixed_loss_metrics: Sequence[str] = tuple(
            make_with_data_set_prefixes(self.loss_metrics)
        )

        self.free_parameter_count_key: str = "free_parameter_count"
        self.masked_parameter_count_key: str = "masked_parameter_count"
        self.fit_number: str = "fit_number"
        self.fit_epoch: str = "fit_epoch"
        self.retained: str = "retained"
        self.seed_number: str = "seed_number"

        def cmax(a):
            # Thanks: https://stackoverflow.com/questions/40672186/cumulative-argmax-of-a-numpy-array
            a = numpy.where(numpy.isnan(a), -numpy.inf, a)
            m = numpy.fmax.accumulate(a)
            x = numpy.arange(a.shape[0])
            x[1:] *= m[:-1] < m[1:]
            numpy.maximum.accumulate(x, axis=0, out=x)
            return m, x

        def cmin(a):
            # Thanks: https://stackoverflow.com/questions/40672186/cumulative-argmax-of-a-numpy-array
            a = numpy.where(numpy.isnan(a), numpy.inf, a)
            m = numpy.fmin.accumulate(a)
            x = numpy.arange(a.shape[0])
            x[1:] *= m[:-1] > m[1:]
            numpy.maximum.accumulate(x, axis=0, out=x)
            return m, x

        # cmin = lambda c: numpy.argmin.accumulate(c)
        imin = lambda c: c.idxmin()
        # cmax = lambda c: c.cummax()
        imax = lambda c: c.idxmax()

        self.run_summary_metrics: Sequence[
            Tuple[str, Callable, Callable, str, str]
        ] = tuple(
            chain(
                *[
                    [
                        [
                            key,
                            cfunc,
                            ifunc,
                            key + "_" + suffix,
                            key + "_" + self.cepoch,
                        ]
                        for key in make_with_prefixes(
                            self.test_data_sets,
                            [metric],
                        )
                    ]
                    for metric, cfunc, ifunc, suffix in chain(
                        [
                            (metric, cmin, imin, self.cumulative_best)
                            for metric in self.loss_metrics
                            if metric != "accuracy"
                        ],
                        [
                            (self.accuracy, cmax, imax, self.cumulative_best),
                        ],  # type: ignore
                    )  # type: ignore
                ]  # type: ignore
            )  # type: ignore
        )  # type: ignore

        self.simple_summarize_keys: Set[str] = set(
            [
                self.epoch_start_time_ms,
                self.epoch,
            ]
            + make_with_data_set_prefixes((self.interval_suffix,))
            + make_with_prefixes(
                [self.train, self.trained, self.validation],
                self.loss_metrics,
            )
            + [
                epoch_column
                for column, cfunc, ifunc, result_column, epoch_column in self.run_summary_metrics
            ]
        )

        test_summary_metrics = [
            (name, cumulative_column, epoch_column)
            for name, _, _, cumulative_column, epoch_column in self.run_summary_metrics
            if name.startswith(self.test)
        ]

        self.quantile_metrics = [
            *[name for name, cumulative_column, epoch_column in test_summary_metrics],
            *[
                cumulative_column
                for name, cumulative_column, epoch_column in test_summary_metrics
            ],
            # *[
            #     epoch_column
            #     for name, cumulative_column, epoch_column in test_summary_metrics
            # ],
        ]


keys = TrainingExperimentKeys()
