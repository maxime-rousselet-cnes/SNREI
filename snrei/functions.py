from typing import Callable, Optional

from cv2 import dilate, erode
from geopandas import GeoDataFrame, clip, sjoin
from numpy import (
    abs,
    arange,
    argsort,
    array,
    concatenate,
    cos,
    expand_dims,
    inf,
    linspace,
    max,
    ndarray,
    newaxis,
    ones,
    pi,
    prod,
    round,
    setdiff1d,
    unique,
    vstack,
    zeros,
)
from numpy.linalg import pinv
from pandas import DataFrame
from pyshtools import SHCoeffs
from scipy import interpolate
from shapely.geometry import Point, Polygon

LAT_LON_PROJECTION = 4326
EARTH_EQUAL_PROJECTION = 3857
MASK_DECIMALS = 5
KERNEL = ones(shape=(3, 3))


def precise_curvature(
    x_initial_values: ndarray,
    f: Callable[[float], ndarray[complex]],
    max_tol: float,
    decimals: int,
) -> tuple[ndarray, ndarray]:
    """
    Finds a sufficiently precise sampling of x axis for f function representation. The criteria takes curvature into account is
    the error between the 1st and second orders.
    """

    # Initializes.
    x_new_values = x_initial_values
    x_values = array(object=[], dtype=float)
    f_values = array(object=[], dtype=complex)

    # Loops while there are still added abscissas.
    while len(x_new_values) != 0:
        # Calls f for new values only.
        f_new_values = f(x_new_values)

        # Updates.
        x_values = concatenate((x_values, x_new_values))
        f_values = f_new_values if len(f_values) == 0 else concatenate((f_values, f_new_values))
        order = argsort(x_values)
        x_values = x_values[order]
        f_values = f_values[order]
        x_new_values = array(object=[], dtype=float)

        # Iterates on new sampling.
        for f_left, f_x, f_right, x_left, x, x_right in zip(
            f_values[:-2],
            f_values[1:-1],
            f_values[2:],
            x_values[:-2],
            x_values[1:-1],
            x_values[2:],
        ):
            # For maximal curvature: finds where the error is above maximum threshold parameter and adds median values.
            condition: ndarray = abs((f_right - f_left) / (x_right - x_left) * (x - x_left) + f_left - f_x) > max_tol * max(
                a=abs([f_left, f_x, f_right]), axis=0
            )
            if condition.any():
                # Updates sampling.
                x_new_values = concatenate((x_new_values, [(x + x_left) / 2.0, (x + x_right) / 2.0]))

        # Keeps only values that are not already taken into account.
        x_new_values = setdiff1d(ar1=unique(round(a=x_new_values, decimals=decimals)), ar2=x_values)

    return x_values, f_values


def interpolate_array(x_values: ndarray, y_values: ndarray, new_x_values: ndarray) -> ndarray:
    """
    1D-Interpolates the given data on its first axis, whatever its shape is.
    """

    # Flattens all other dimensions.
    shape = y_values.shape
    y_values.reshape((shape[0], -1))
    components = y_values.shape[1]

    # Initializes
    function_values = zeros(shape=(len(new_x_values), components), dtype=complex)

    # Loops on components
    for i_component, component in enumerate(y_values.transpose()):

        # Creates callable (linear).
        function = interpolate.interp1d(x=x_values, y=component, kind="linear")

        # Calls linear interpolation on new x values.
        function_values[:, i_component] = function(x=new_x_values)

    #  Converts back into initial other dimension shapes.
    function_values.reshape((len(new_x_values), *shape[1:]))
    return function_values


def interpolate_all(
    x_values_per_component: list[ndarray],
    function_values: list[ndarray],
    x_shared_values: ndarray,
) -> ndarray:
    """
    Interpolate several function values on shared abscissas.
    """
    return array(
        object=(
            function_values
            if len(x_shared_values) == 1 and x_shared_values[0] == inf  # Manages elastic case.
            else [
                interpolate_array(
                    x_values=x_tab,
                    y_values=function_values_tab,
                    new_x_values=x_shared_values,
                )
                for x_tab, function_values_tab in zip(x_values_per_component, function_values)
            ]
        )
    )


def get_degrees_indices(degrees: list[int], degrees_to_plot: list[int]) -> list[int]:
    """
    Returns the indices of the wanted degrees in the list of degrees.
    """
    return [list(degrees).index(degree) for degree in degrees_to_plot]


def signal_trend(trend_dates: ndarray[float], signal: ndarray[float]) -> tuple[float, float]:
    """
    Returns signal's trend: mean slope and additive constant during last years (LSE).
    """
    # Assemble matrix A.
    A = vstack(
        [
            trend_dates,
            ones(len(trend_dates)),
        ]
    ).T
    # Direct least square regression using pseudo-inverse.
    result: ndarray = pinv(A).dot(signal[:, newaxis])
    return result.flatten()  # Turn the signal into a column vector. (slope, additive_constant)


def map_normalizing(
    map: ndarray,
) -> ndarray:
    """
    Sets global mean as zero and max as one by homothety.
    """
    n_t = prod(map.shape)
    sum_map = sum(map.flatten())
    max_map = max(map.flatten())
    return map / (max_map - sum_map / n_t) + sum_map / (sum_map - max_map * n_t)


def surface_ponderation(
    mask: ndarray[float],
) -> ndarray[float]:
    """
    Gets the surface of a (latitude * longitude) array.
    """
    return mask * expand_dims(a=cos(linspace(start=-pi / 2, stop=pi / 2, num=len(mask))), axis=1)


def mean_on_mask(
    mask: ndarray[float],
    harmonics: Optional[ndarray[float]] = None,
    grid: Optional[ndarray[float]] = None,
) -> float:
    """
    Computes mean value over a given surface. Uses a given mask.
    """
    if grid is None:
        grid: ndarray[float] = make_grid(harmonics=harmonics)
    surface = surface_ponderation(mask=mask)
    weighted_values = grid * surface
    return round(a=sum(weighted_values.flatten()) / sum(surface.flatten()), decimals=MASK_DECIMALS)


def make_grid(harmonics: ndarray[float]) -> ndarray[float]:
    """ """
    return SHCoeffs.from_array(harmonics).expand(extend=False).to_array()


def build_ocean_mask(continents: GeoDataFrame, n_max: int) -> ndarray[float]:
    """"""
    lat = list(90 - arange(2 * (n_max + 1)))
    lon = list(180 - arange(4 * (n_max + 1)))
    ocean_mask = ones(shape=(len(lat), len(lon)))
    for centroid in continents.geometry.centroid:
        ocean_mask[lat.index(round(centroid.y)), (180 - lon.index(round(centroid.x)) % 360)] = 0.0
    return dilate(erode(ocean_mask, kernel=KERNEL), kernel=KERNEL)


def geopandas_oceanic_mean(
    continents: GeoDataFrame,
    latitudes: ndarray[float],
    longitudes: ndarray[float],
    harmonics: Optional[ndarray[float]] = None,
    grid: Optional[ndarray[float]] = None,
) -> float:
    """"""
    if grid is None:
        grid: ndarray[float] = make_grid(harmonics=harmonics)
        n_max = harmonics.shape[1] - 1
    else:
        n_max = len(grid) // 2 - 1
    grid_gdf = generate_grid(EWH=grid, latitudes=latitudes, longitudes=longitudes, n_max=n_max)
    # Performs a spatial join to identify points on land or near land.
    land_gdf = sjoin(grid_gdf, continents[continents["EWH"] == 1.0], predicate="intersects")

    # Identifies oceanic points by selecting points not in land_gdf.
    oceanic_gdf = grid_gdf.loc[~grid_gdf.index.isin(land_gdf.index)]

    # Computes the mean EWH value over the oceanic points.
    return oceanic_gdf["EWH"].mean()


def generate_grid(
    EWH: ndarray[float],
    latitudes: ndarray[float],
    longitudes: ndarray[float],
    n_max: int,
):
    """"""
    # Adjust longitude in EWH to shift from 0-360 to -180 to 180
    longitudes = [(lon - 360 if lon > 180 else lon) for lon in longitudes]

    # Converts EWH to a DataFrame with adjusted longitude.
    ewh_df = DataFrame(EWH, columns=longitudes, index=latitudes)
    ewh_df = ewh_df.stack().reset_index()
    ewh_df.columns = ["latitude", "longitude", "EWH"]

    # Converts to GeoDataFrame.
    ewh_df["geometry"] = ewh_df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
    ewh_gdf = GeoDataFrame(ewh_df, geometry="geometry")
    ewh_gdf.crs = LAT_LON_PROJECTION

    # Step 2: Create a grid of polygons.
    polygons = []
    values = []
    lon_len = 4 * (n_max + 1)

    for i_line in range(1, 2 * n_max + 1):
        for i_column in range(lon_len - 1):
            # Gets the current points.
            p1 = ewh_gdf.iloc[(lon_len * i_line) + ((2 * n_max + 2 + i_column) % lon_len)].geometry
            p2 = ewh_gdf.iloc[(lon_len * (i_line + 1)) + ((2 * n_max + 2 + i_column) % lon_len)].geometry
            p3 = ewh_gdf.iloc[(lon_len * (i_line + 1)) + ((2 * n_max + 2 + i_column + 1) % lon_len)].geometry
            p4 = ewh_gdf.iloc[(lon_len * i_line) + ((2 * n_max + 2 + i_column + 1) % lon_len)].geometry

            # Creates a polygon using the four points.
            polygon = Polygon([p1, p2, p3, p4])
            polygons.append(polygon)
            values.append(ewh_gdf.iloc[lon_len * i_line + (2 * n_max + 3 + i_column) % lon_len]["EWH"])

    # Step 3: Creates a new GeoDataFrame with the polygons.
    return GeoDataFrame({"EWH": values}, geometry=polygons, crs=ewh_gdf.crs)


def generate_continents_buffered_reprojected_grid(
    EWH: ndarray[float],
    latitudes: ndarray[float],
    longitudes: ndarray[float],
    n_max: int,
    continents: GeoDataFrame,
    buffer_distance: float,
):
    """ """
    grid = generate_grid(EWH=EWH, latitudes=latitudes, longitudes=longitudes, n_max=n_max)

    # Projects to an earth equal projection CRS to bufferize.
    continents_projected = continents.to_crs(crs=EARTH_EQUAL_PROJECTION)
    grid_projected = grid.to_crs(EARTH_EQUAL_PROJECTION)
    continents_buffered: GeoDataFrame = continents_projected.copy()
    continents_buffered["geometry"] = continents_buffered.buffer(buffer_distance * 1e3)  # (km) -> (m).
    continents_clipped = clip(grid_projected, continents_buffered)

    # Gets base projection.
    return continents_clipped.to_crs(crs=LAT_LON_PROJECTION)
