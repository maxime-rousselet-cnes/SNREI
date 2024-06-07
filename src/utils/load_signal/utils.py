from numpy import (
    arange,
    array,
    ceil,
    concatenate,
    flip,
    linspace,
    log2,
    ndarray,
    round,
    tensordot,
    where,
    zeros,
)
from scipy import interpolate
from scipy.fft import fft, fftfreq

from ...functions import (
    map_normalizing,
    mean_on_mask,
    signal_trend,
    surface_ponderation,
)
from ..classes import LoadSignalHyperParameters
from ..data import (
    extract_GRACE_data,
    extract_mask_csv,
    extract_mask_nc,
    extract_temporal_load_signal,
    get_ocean_mask,
    map_sampling,
)


def get_trend_dates(
    signal_dates: ndarray[float] | list[float],
    load_signal_hyper_parameters: LoadSignalHyperParameters,
) -> tuple[ndarray[float], ndarray[int]]:
    """
    Returns trend indices and trend dates.
    """
    shift_dates = (
        array(object=signal_dates, dtype=float)
        + load_signal_hyper_parameters.spline_time_years
        + load_signal_hyper_parameters.last_year_for_trend
    )
    trend_indices = where(
        (shift_dates <= load_signal_hyper_parameters.last_year_for_trend - 1)
        * (shift_dates >= load_signal_hyper_parameters.first_year_for_trend)
    )[0]
    return trend_indices, shift_dates[trend_indices]


def build_elastic_load_signal_history(
    initial_signal_dates: ndarray[float],
    load_signal: ndarray[float],
    load_signal_hyper_parameters: LoadSignalHyperParameters,
) -> tuple[ndarray[float], ndarray[float], float, float]:
    """
    Builds a load signal history function suitable for Fourier analysis. It has zero mean value, antisymetry and no Gibbs
    effect.
    Returns dates, signal, time step and trend.
    """

    # Linearly extends the signal for last years.
    trend_indices = initial_signal_dates >= load_signal_hyper_parameters.first_year_for_trend
    elastic_load_signal_trend, elastic_load_signal_additive_constant = signal_trend(
        trend_dates=initial_signal_dates[trend_indices],
        signal=load_signal[trend_indices],
    )
    extend_part_dates = arange(initial_signal_dates[-1] + 1, load_signal_hyper_parameters.last_year_for_trend + 1)
    extend_part_load_signal = elastic_load_signal_trend * extend_part_dates + elastic_load_signal_additive_constant

    # Eventually includes a LIA effect.
    if load_signal_hyper_parameters.LIA:
        LIA_value = load_signal[-1] * load_signal_hyper_parameters.LIA_amplitude_effect
        # zero plateau after LIA.
        load_signal = (
            concatenate(
                (
                    linspace(
                        start=LIA_value,
                        stop=0.0,
                        num=load_signal_hyper_parameters.LIA_time_years,
                    ),
                    zeros(shape=(initial_signal_dates[0] - load_signal_hyper_parameters.LIA_end_date)),
                    load_signal,
                )
            )
            - LIA_value
        )
        extend_part_load_signal -= LIA_value

    # Creates cubic spline for antisymetry.
    mean_slope = extend_part_load_signal[-1] / load_signal_hyper_parameters.spline_time_years
    spline = (
        lambda T: mean_slope / (2.0 * load_signal_hyper_parameters.spline_time_years**2.0) * T**3.0 - 3.0 * mean_slope / 2 * T
    )

    # Builds signal history / Creates an initial plateau at zero value.
    extended_time_serie_past = concatenate(
        (
            zeros(shape=(load_signal_hyper_parameters.initial_plateau_time_years)),
            load_signal,
            extend_part_load_signal,
            spline(T=arange(start=-load_signal_hyper_parameters.spline_time_years, stop=0)),
        )
    )

    # Applies antisymetry.
    extended_time_serie = concatenate((extended_time_serie_past, [0], -flip(m=extended_time_serie_past)))

    # Deduces dates axis.
    n_extended_signal = len(extended_time_serie)
    extended_dates = arange(stop=n_extended_signal) - n_extended_signal // 2

    # Interpolates at sufficient sampling for no Gibbs effect.
    n_log_min_no_Gibbs = round(ceil(log2(n_extended_signal)))
    half_signal_period = max(extended_dates)
    n_signal = int(2 ** (n_log_min_no_Gibbs + load_signal_hyper_parameters.anti_Gibbs_effect_factor))
    signal_dates = linspace(-half_signal_period, stop=half_signal_period, num=n_signal)

    return (
        signal_dates,
        interpolate.splev(x=signal_dates, tck=interpolate.splrep(x=extended_dates, y=extended_time_serie, k=3)),  # Signal.
        2.0 * half_signal_period / n_signal,  # Time step.
        elastic_load_signal_trend,  # Trend.
    )


def build_frequencial_elastic_load_signal(
    load_signal_hyper_parameters: LoadSignalHyperParameters,
) -> tuple[ndarray[float], ndarray[float], ndarray[complex], float]:
    """
    Builds the load in the frequential domain.
    Returns:
        - dates (yr)
        - frequencies (yr^-1)
        - frequential elastic load signal (mm)
        - elastic load signal trend (mm/yr)
    """
    # Builds the signal's frequencial component.
    dates, load_signal = extract_temporal_load_signal(
        name=load_signal_hyper_parameters.case, filename=load_signal_hyper_parameters.load_history
    )
    signal_dates, temporal_elastic_load_signal, time_step, elastic_load_signal_trend = build_elastic_load_signal_history(
        signal_dates=dates,
        load_signal=load_signal,
        load_signal_hyper_parameters=load_signal_hyper_parameters,
    )  # (yr).
    elastic_load_signal_frequencial_component = fft(x=temporal_elastic_load_signal)
    frequencies = fftfreq(n=len(elastic_load_signal_frequencial_component), d=time_step)

    return (
        signal_dates,
        frequencies,
        elastic_load_signal_frequencial_component,
        elastic_load_signal_trend,
    )


def build_frequencial_harmonic_elastic_load_signal(
    load_signal_hyper_parameters: LoadSignalHyperParameters,
) -> tuple[ndarray[float], ndarray[float], ndarray[complex], ndarray[float], float]:
    """
    Builds the load in the frequential-harmonic domain.
    Returns:
        - dates (yr)
        - frequencies (yr^-1)
        - frequential elastic load signal (4-D array: C/S, n, m, frequency) (mm)
        - elastic load signal trend (3-D array: C/S, n, m) (mm/yr)
    """

    # For Frederikse/GRACE trends data.
    if load_signal_hyper_parameters.load_signal == "load_history":

        # Gets frequencial component.
        (
            signal_dates,
            frequencies,
            elastic_load_signal_frequencial_component,
            elastic_load_signal_trend,
        ) = build_frequencial_elastic_load_signal(load_signal_hyper_parameters=load_signal_hyper_parameters)

        # Gets harmonic component.
        if load_signal_hyper_parameters.load_spatial_behaviour_data == "GRACE":
            map = extract_GRACE_data(
                name=load_signal_hyper_parameters.load_spatial_behaviour_file, skiprows=load_signal_hyper_parameters.skiprows
            )
        else:  # Considered as ocean/land repartition only.
            map = map_normalizing(
                map=(
                    extract_mask_csv(name=load_signal_hyper_parameters.load_spatial_behaviour_file)
                    if load_signal_hyper_parameters.load_spatial_behaviour_file.split(".")[-1] == "csv"
                    else extract_mask_nc(name=load_signal_hyper_parameters.load_spatial_behaviour_file)
                )
            )
        # Loads the continents with opposite value, such that global mean is null.
        if load_signal_hyper_parameters.opposite_load_on_continents:
            ocean_mask = get_ocean_mask(name=load_signal_hyper_parameters.ocean_mask, n_max=(len(map) - 1) // 2)
            map = map * ocean_mask - (1.0 - ocean_mask) * (
                mean_on_mask(grid=map, mask=ocean_mask)
                * sum(surface_ponderation(mask=ocean_mask).flatten())
                / sum(surface_ponderation(mask=(1.0 - ocean_mask)).flatten())
            )

        spatial_component = map_sampling(map=map, n_max=load_signal_hyper_parameters.n_max, harmonic_domain=True)[0]

        # Projects.
        return (
            signal_dates,
            frequencies,
            tensordot(a=spatial_component, b=elastic_load_signal_frequencial_component, axes=0) / elastic_load_signal_trend,
            spatial_component,
        )

    else:
        # TODO: Get GRACE's data history ?
        pass
