import io, os, time, warnings, contextlib, tempfile, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from scipy.special import jv
from scipy.optimize import least_squares
warnings.filterwarnings("ignore")

# Solver settings retained from the supplied implementation
RANDOM_SEED = 12345
IS_BONDED = True
NUMBER_OF_STARTS = 5
CACHE_DECIMAL_PLACES = 12
MAX_NFEV = 200
MAX_BOUND_EXPANSIONS = 100
BOUND_EXPANSION_MPA = 1000.0
USE_PROGRESSIVE_BOUND_EXPANSION = True
LOCAL_WINDOW_HALF_WIDTH_MPA = 10.0
LOCAL_WINDOW_GROWTH_FACTOR = 2.0
HISTORY_DEDUP_RTOL = 1e-9
HISTORY_DEDUP_ATOL = 1e-8
BOUND_TOLERANCE = 0.02
ABSOLUTE_MIN_MODULUS = 1.0
ABSOLUTE_MAX_MODULUS = 50000.0

# Optional Streamlit callback used to render live solver progress.
LIVE_PROGRESS_CALLBACK = None


class _TargetRMSEReached(Exception):
    """Internal signal to stop optimization when the requested RMSE is met."""
    def __init__(self, E, calculated, error, rmse):
        super().__init__("User-entered target RMSE reached.")
        self.E = np.asarray(E, dtype=float).copy()
        self.calculated = np.asarray(calculated, dtype=float).copy()
        self.error = np.asarray(error, dtype=float).copy()
        self.rmse = float(rmse)

def r_hat(

    n,

    thickness,

    E,

    v,

    isbonded,

    m,

    points

):


    # ============================================================
    # LAYER DEPTHS
    # ============================================================

    z = np.cumsum(
        thickness
    )


    H = z[-1]


    # ============================================================
    # LAMBDA VALUES
    # ============================================================

    lambda_values = np.zeros(
        n
    )


    for i in range(n - 1):

        lambda_values[i] = (

            z[i]
            /
            H

        )


    lambda_values[n - 1] = np.inf


    # ============================================================
    # F AND R
    # ============================================================

    F = np.zeros(
        n
    )


    R = np.zeros(
        n - 1
    )


    F[0] = np.exp(

        -m
        *
        lambda_values[0]

    )


    R[0] = (

        E[0]
        /
        E[1]

        *

        (1 + v[1])
        /
        (1 + v[0])

    )


    for i in range(

        1,
        n - 1

    ):


        F[i] = np.exp(

            -m

            *

            (

                lambda_values[i]

                -

                lambda_values[i - 1]

            )

        )


        R[i] = (

            E[i]
            /
            E[i + 1]

            *

            (1 + v[i + 1])
            /
            (1 + v[i])

        )


    F[n - 1] = 0.0


    # ============================================================
    # INTERFACE MATRICES
    # ============================================================

    M_list = []


    for i in range(n - 1):


        M1 = np.zeros(
            (4, 4)
        )


        M2 = np.zeros(
            (4, 4)
        )


        # ========================================================
        # BONDED
        # ========================================================

        if isbonded:


            M1[0, 0] = 1
            M1[1, 0] = 1
            M1[2, 0] = 1
            M1[3, 0] = 1


            M1[0, 1] = F[i]
            M1[1, 1] = -F[i]
            M1[2, 1] = F[i]
            M1[3, 1] = -F[i]


            M1[0, 2] = (

                -(
                    1
                    -
                    2 * v[i]
                    -
                    m * lambda_values[i]
                )

            )


            M1[1, 2] = (

                2 * v[i]
                +
                m * lambda_values[i]

            )


            M1[2, 2] = (

                1
                +
                m * lambda_values[i]

            )


            M1[3, 2] = (

                -(
                    2
                    -
                    4 * v[i]
                    -
                    m * lambda_values[i]
                )

            )


            M1[0, 3] = (

                1
                -
                2 * v[i]
                +
                m * lambda_values[i]

            ) * F[i]


            M1[1, 3] = (

                2 * v[i]
                -
                m * lambda_values[i]

            ) * F[i]


            M1[2, 3] = (

                -(
                    1
                    -
                    m * lambda_values[i]
                )

                *

                F[i]

            )


            M1[3, 3] = (

                -(
                    2
                    -
                    4 * v[i]
                    +
                    m * lambda_values[i]
                )

                *

                F[i]

            )


            # ====================================================
            # M2
            # ====================================================

            M2[0, 0] = F[i + 1]

            M2[1, 0] = F[i + 1]


            M2[2, 0] = (

                R[i]
                *
                F[i + 1]

            )


            M2[3, 0] = (

                R[i]
                *
                F[i + 1]

            )


            M2[0, 1] = 1

            M2[1, 1] = -1


            M2[2, 1] = R[i]

            M2[3, 1] = -R[i]


            M2[0, 2] = (

                -(
                    1
                    -
                    2 * v[i + 1]
                    -
                    m * lambda_values[i]
                )

                *

                F[i + 1]

            )


            M2[1, 2] = (

                2 * v[i + 1]
                +
                m * lambda_values[i]

            ) * F[i + 1]


            M2[2, 2] = (

                (
                    1
                    +
                    m * lambda_values[i]
                )

                *

                R[i]

                *

                F[i + 1]

            )


            M2[3, 2] = (

                -(
                    2
                    -
                    4 * v[i + 1]
                    -
                    m * lambda_values[i]
                )

                *

                R[i]

                *

                F[i + 1]

            )


            M2[0, 3] = (

                1
                -
                2 * v[i + 1]
                +
                m * lambda_values[i]

            )


            M2[1, 3] = (

                2 * v[i + 1]
                -
                m * lambda_values[i]

            )


            M2[2, 3] = (

                -(
                    1
                    -
                    m * lambda_values[i]
                )

                *

                R[i]

            )


            M2[3, 3] = (

                -(
                    2
                    -
                    4 * v[i + 1]
                    +
                    m * lambda_values[i]
                )

                *

                R[i]

            )


        # ========================================================
        # UNBONDED
        # ========================================================

        else:


            M1[0, 0] = 1
            M1[1, 0] = 1
            M1[2, 0] = 1
            M1[3, 0] = 0


            M1[0, 1] = F[i]
            M1[1, 1] = F[i]
            M1[2, 1] = -F[i]
            M1[3, 1] = 0


            M1[0, 2] = (

                -(
                    1
                    -
                    2 * v[i]
                    -
                    m * lambda_values[i]
                )

            )


            M1[1, 2] = (

                1
                +
                m * lambda_values[i]

            )


            M1[2, 2] = (

                2 * v[i]
                +
                m * lambda_values[i]

            )


            M1[3, 2] = 0


            M1[0, 3] = (

                1
                -
                2 * v[i]
                +
                m * lambda_values[i]

            ) * F[i]


            M1[1, 3] = (

                -(
                    1
                    -
                    m * lambda_values[i]
                )

                *

                F[i]

            )


            M1[2, 3] = (

                2 * v[i]
                -
                m * lambda_values[i]

            ) * F[i]


            M1[3, 3] = 0


            M2[0, 0] = F[i + 1]


            M2[1, 0] = (

                R[i]
                *
                F[i + 1]

            )


            M2[2, 0] = 0


            M2[3, 0] = F[i + 1]


            M2[0, 1] = 1


            M2[1, 1] = R[i]


            M2[2, 1] = 0


            M2[3, 1] = -1


            M2[0, 2] = (

                -(
                    1
                    -
                    2 * v[i + 1]
                    -
                    m * lambda_values[i]
                )

                *

                F[i + 1]

            )


            M2[1, 2] = (

                (
                    1
                    +
                    m * lambda_values[i]
                )

                *

                R[i]

                *

                F[i + 1]

            )


            M2[2, 2] = 0


            M2[3, 2] = (

                2 * v[i + 1]
                +
                m * lambda_values[i]

            ) * F[i + 1]


            M2[0, 3] = (

                1
                -
                2 * v[i + 1]
                +
                m * lambda_values[i]

            )


            M2[1, 3] = (

                -(
                    1
                    -
                    m * lambda_values[i]
                )

                *

                R[i]

            )


            M2[2, 3] = 0


            M2[3, 3] = (

                2 * v[i + 1]
                -
                m * lambda_values[i]

            )


        # ========================================================
        # MATRIX SOLUTION
        # ========================================================

        try:


            Mi = np.linalg.solve(

                M1,

                M2

            )


        except np.linalg.LinAlgError:


            Mi = (

                np.linalg.pinv(M1)

                @

                M2

            )


        M_list.append(
            Mi
        )


    # ============================================================
    # MATRIX PRODUCT
    # ============================================================

    MM = np.eye(4)


    for Mi in M_list:

        MM = (

            MM
            @
            Mi

        )


    MM = MM[:, [1, 3]]


    # ============================================================
    # BOUNDARY CONDITIONS
    # ============================================================

    b11 = np.exp(

        -lambda_values[0]
        *
        m

    )


    b21 = np.exp(

        -lambda_values[0]
        *
        m

    )


    b12 = 1.0

    b22 = -1.0


    c11 = (

        -(1 - 2 * v[0])

        *

        np.exp(

            -m
            *
            lambda_values[0]

        )

    )


    c21 = (

        2 * v[0]

        *

        np.exp(

            -m
            *
            lambda_values[0]

        )

    )


    c12 = 1 - 2 * v[0]

    c22 = 2 * v[0]


    k11 = (

        b11 * MM[0, 0]
        +
        b12 * MM[1, 0]
        +
        c11 * MM[2, 0]
        +
        c12 * MM[3, 0]

    )


    k12 = (

        b11 * MM[0, 1]
        +
        b12 * MM[1, 1]
        +
        c11 * MM[2, 1]
        +
        c12 * MM[3, 1]

    )


    k21 = (

        b21 * MM[0, 0]
        +
        b22 * MM[1, 0]
        +
        c21 * MM[2, 0]
        +
        c22 * MM[3, 0]

    )


    k22 = (

        b21 * MM[0, 1]
        +
        b22 * MM[1, 1]
        +
        c21 * MM[2, 1]
        +
        c22 * MM[3, 1]

    )


    determinant = (

        k11 * k22
        -
        k12 * k21

    )


    if abs(determinant) < 1e-14:

        raise np.linalg.LinAlgError(

            "Singular boundary matrix."

        )


    # ============================================================
    # COEFFICIENTS
    # ============================================================

    A = np.zeros(n)

    B = np.zeros(n)

    C = np.zeros(n)

    D = np.zeros(n)


    A[n - 1] = 0.0


    B[n - 1] = (

        k22
        /
        determinant

    )


    C[n - 1] = 0.0


    if abs(k21) > 1e-14:


        D[n - 1] = (

            1.0

            /

            (

                k12

                -

                k22
                *
                k11
                /
                k21

            )

        )


    else:

        D[n - 1] = 0.0


    # ============================================================
    # BACK SUBSTITUTION
    # ============================================================

    for i in range(

        n - 2,
        -1,
        -1

    ):


        X = (

            M_list[i]

            @

            np.array(

                [

                    A[i + 1],

                    B[i + 1],

                    C[i + 1],

                    D[i + 1]

                ]

            )

        )


        A[i] = X[0]

        B[i] = X[1]

        C[i] = X[2]

        D[i] = X[3]


    # ============================================================
    # DISPLACEMENT
    # ============================================================

    results = []


    for pt in points:


        rho = (

            pt["r"]
            /
            H

        )


        lmm = (

            pt["z"]
            /
            H

        )


        ii = (

            pt["layer"]
            -
            1

        )


        if ii != 0:


            exp1 = np.exp(

                -m

                *

                (

                    lambda_values[ii]

                    -

                    lmm

                )

            )


            exp2 = np.exp(

                -m

                *

                (

                    lmm

                    -

                    lambda_values[ii - 1]

                )

            )


        else:


            exp1 = np.exp(

                -m

                *

                (

                    lambda_values[ii]

                    -

                    lmm

                )

            )


            exp2 = np.exp(

                -m
                *
                lmm

            )


        w = (

            -H

            *

            (1 + v[ii])

            /

            E[ii]

            *

            jv(

                0,

                m
                *
                rho

            )

            *

            (

                (

                    A[ii]

                    -

                    C[ii]

                    *

                    (

                        2

                        -

                        4 * v[ii]

                        -

                        m * lmm

                    )

                )

                *

                exp1

                -

                (

                    B[ii]

                    +

                    D[ii]

                    *

                    (

                        2

                        -

                        4 * v[ii]

                        +

                        m * lmm

                    )

                )

                *

                exp2

            )

        )


        results.append(

            {

                "w":
                    w

            }

        )


    return results


# ================================================================
# 6. FORWARD CALCULATION
# ================================================================

def calculate_deflections(

    thickness,

    E

):


    z = np.cumsum(
        thickness
    )


    H = z[-1]


    alpha = (

        LOAD_RADIUS_MM
        /
        H

    )


    dl = (

        0.1
        /
        alpha

    )


    deflections = np.zeros(

        len(POINTS)

    )


    # ============================================================
    # GAUSS INTEGRATION POINTS
    # ============================================================

    gauss_points = [

        (-0.86114, 0.34786),

        (-0.33998, 0.65215),

        (0.33998, 0.65215),

        (0.86114, 0.34786)

    ]


    l = 0.0


    while l <= 20.0 + 1e-12:


        for node, weight in gauss_points:


            m = (

                l

                +

                dl / 2.0

                +

                node
                *
                dl
                /
                2.0

            )


            fc = (

                weight

                *

                dl

                /

                2.0

            )


            if m <= 1e-14:

                continue


            try:


                rhat_results = r_hat(

                    N,

                    thickness,

                    E,

                    POISSON,

                    IS_BONDED,

                    m,

                    POINTS

                )


            except Exception:


                return np.full(

                    len(POINTS),

                    np.nan

                )


            scale = (

                fc

                *

                PRESSURE_MPA

                *

                alpha

                *

                jv(

                    1,

                    m
                    *
                    alpha

                )

                /

                m

            )


            for j in range(

                len(POINTS)

            ):


                deflections[j] += (

                    scale

                    *

                    rhat_results[j]["w"]

                )


        l += dl


    return deflections


# ================================================================
# 7. CALCULATE R-SQUARED
# ================================================================

def calculate_r2(

    target,

    calculated

):


    target = np.asarray(

        target,

        dtype=float

    )


    calculated = np.asarray(

        calculated,

        dtype=float

    )


    ss_res = np.sum(

        (

            target
            -
            calculated

        )

        ** 2

    )


    ss_tot = np.sum(

        (

            target
            -
            np.mean(target)

        )

        ** 2

    )


    if ss_tot <= 1e-30:

        return np.nan


    return (

        1.0

        -

        ss_res
        /
        ss_tot

    )


# ================================================================
# 8. CHECK BOUND HIT
# ================================================================

def check_bound_hit(

    E,

    lower_bounds,

    upper_bounds,

    tolerance

):


    E = np.asarray(

        E,

        dtype=float

    )


    lower_bounds = np.asarray(

        lower_bounds,

        dtype=float

    )


    upper_bounds = np.asarray(

        upper_bounds,

        dtype=float

    )


    current_range = (

        upper_bounds
        -
        lower_bounds

    )


    lower_threshold = (

        lower_bounds

        +

        tolerance
        *
        current_range

    )


    upper_threshold = (

        upper_bounds

        -

        tolerance
        *
        current_range

    )


    hit_lower = (

        E <= lower_threshold

    )


    hit_upper = (

        E >= upper_threshold

    )


    return (

        hit_lower,

        hit_upper

    )


# ================================================================
# 9. EXPAND BOUNDS
# ================================================================
#
# IMPORTANT:
# This function expands ONLY the modulus that actually hits a bound.
#
# Example:
#     Current E1 = 2000 MPa at upper bound
#     BOUND_EXPANSION_MPA = 1000
#     New E1 upper bound = 3000 MPa
#
# The other modulus bounds are handled separately by
# create_next_stage_bounds(), where their previous recovered
# values are protected inside a local window.
# ================================================================

def expand_bounds(
    lower_bounds,
    upper_bounds,
    hit_lower,
    hit_upper
):

    lower_bounds = np.asarray(
        lower_bounds,
        dtype=float
    )

    upper_bounds = np.asarray(
        upper_bounds,
        dtype=float
    )

    new_lower = lower_bounds.copy()
    new_upper = upper_bounds.copy()

    for i in range(len(lower_bounds)):

        # ----------------------------------------------------------
        # EXPAND LOWER BOUND
        # ----------------------------------------------------------

        if hit_lower[i]:

            interval = max(
                upper_bounds[i] - lower_bounds[i],
                BOUND_EXPANSION_MPA
            )
            expansion = (
                max(BOUND_EXPANSION_MPA, interval)
                if USE_PROGRESSIVE_BOUND_EXPANSION
                else BOUND_EXPANSION_MPA
            )
            new_lower[i] = max(
                ABSOLUTE_MIN_MODULUS,
                lower_bounds[i] - expansion
            )

        # ----------------------------------------------------------
        # EXPAND UPPER BOUND
        # ----------------------------------------------------------

        if hit_upper[i]:

            interval = max(
                upper_bounds[i] - lower_bounds[i],
                BOUND_EXPANSION_MPA
            )
            expansion = (
                max(BOUND_EXPANSION_MPA, interval)
                if USE_PROGRESSIVE_BOUND_EXPANSION
                else BOUND_EXPANSION_MPA
            )
            new_upper[i] = min(
                ABSOLUTE_MAX_MODULUS,
                upper_bounds[i] + expansion
            )

    return (
        new_lower,
        new_upper
    )


# ================================================================
# 9A. CREATE NEXT-STAGE BOUNDS
# ================================================================
#
# This is the key change for your idea.
#
# Suppose:
#
#     Stage result:
#         E1 = 2000 MPa  -> upper bound hit
#         E2 = 70 MPa    -> correct / not at bound
#         E3 = 20 MPa    -> correct / not at bound
#
# Then the next stage becomes approximately:
#
#     E1 =  ... existing lower bound to 3000 MPa
#     E2 = 60 to 80 MPa
#     E3 = 10 to 30 MPa
#
# This prevents E2 and E3 from freely jumping back to their
# original large search ranges when E1 is expanded.
# ================================================================

def create_next_stage_bounds(
    current_lower_bounds,
    current_upper_bounds,
    best_E,
    hit_lower,
    hit_upper
):
    """
    Create the next search region.

    Only bounds that were actually hit are expanded:
      * lower-bound hit: lower bound decreases by 1000 MPa
      * upper-bound hit: upper bound increases by 1000 MPa

    Every modulus that did not hit a bound retains its complete
    current range unchanged. No +/-10 MPa local window is applied.
    """
    current_lower_bounds = np.asarray(current_lower_bounds, dtype=float)
    current_upper_bounds = np.asarray(current_upper_bounds, dtype=float)
    hit_lower = np.asarray(hit_lower, dtype=bool)
    hit_upper = np.asarray(hit_upper, dtype=bool)

    next_lower = current_lower_bounds.copy()
    next_upper = current_upper_bounds.copy()

    for i in range(len(next_lower)):
        if hit_lower[i]:
            next_lower[i] = max(
                ABSOLUTE_MIN_MODULUS,
                current_lower_bounds[i] - BOUND_EXPANSION_MPA
            )

        if hit_upper[i]:
            next_upper[i] = min(
                ABSOLUTE_MAX_MODULUS,
                current_upper_bounds[i] + BOUND_EXPANSION_MPA
            )

    return next_lower, next_upper


# ================================================================
# 10. CREATE STARTING POINTS
# ================================================================

def get_start_points(

    lower_bounds,

    upper_bounds,

    previous_best,

    stage_number

):


    rng = np.random.default_rng(

        RANDOM_SEED

        +

        stage_number
        *
        7919

    )


    starts = []


    # ============================================================
    # CENTER POINT
    # ============================================================

    center = (

        lower_bounds
        +
        upper_bounds

    ) / 2.0


    starts.append(
        center
    )


    # ============================================================
    # PREVIOUS BEST POINT
    # ============================================================

    if previous_best is not None:


        previous_start = np.clip(

            previous_best,

            lower_bounds,

            upper_bounds

        )


        starts.append(
            previous_start
        )


    # ============================================================
    # RANDOM STARTING POINTS
    # ============================================================

    while len(starts) < NUMBER_OF_STARTS:


        random_start = rng.uniform(

            lower_bounds,

            upper_bounds

        )


        starts.append(
            random_start
        )


    return starts


# ================================================================
# 11. ADAPTIVE BACKWARD SOLVER
# ================================================================

def adaptive_backward_solver(


    thickness,

    target,

    initial_lower_bounds,

    initial_upper_bounds,

    target_rmse=None


):
    # Use the threshold supplied for THIS run, rather than relying only on a
    # module global that can become stale during Streamlit reruns or batch rows.
    if target_rmse is None:
        target_rmse = float(globals().get("TARGET_RMSE", 0.0001))
    else:
        target_rmse = float(target_rmse)
    if not np.isfinite(target_rmse) or target_rmse <= 0:
        raise ValueError("Target RMSE must be a finite positive number in mm.")


    # ============================================================
    # INPUT CONVERSION
    # ============================================================

    thickness = np.asarray(

        thickness,

        dtype=float

    )


    target = np.asarray(

        target,

        dtype=float

    )


    lower_bounds = np.asarray(

        initial_lower_bounds,

        dtype=float

    ).copy()


    upper_bounds = np.asarray(

        initial_upper_bounds,

        dtype=float

    ).copy()


    # ============================================================
    # HISTORY
    # ============================================================

    history = []


    evaluation_number = 0


    # ============================================================
    # GLOBAL BEST
    # ============================================================

    global_best_E = None

    global_best_calculated = None

    global_best_rmse = np.inf

    global_best_optimizer = None

    global_best_lower_bounds = None

    global_best_upper_bounds = None


    # ============================================================
    # EXPANSION LOOP
    # ============================================================

    for stage in range(

        MAX_BOUND_EXPANSIONS + 1

    ):


        print()

        print("=" * 80)

        print(

            f"BOUND EXPANSION STAGE {stage}"

        )

        print("=" * 80)


        print()

        print(
            "Current Modulus Bounds:"
        )


        for i in range(N):


            print(

                f"E{i + 1}: "

                f"{lower_bounds[i]:.4f}"

                f" to "

                f"{upper_bounds[i]:.4f}"

                f" MPa"

            )


        # ========================================================
        # STAGE BEST
        # ========================================================

        stage_best_E = None

        stage_best_calculated = None

        stage_best_rmse = np.inf

        stage_best_optimizer = None

        # Last modulus vector written to the history for this stage.
        # This prevents repeated rows such as 2250, 2250, 2250, 2250.
        last_history_E = None

        # Per-stage cache: key = rounded modulus vector, value =
        # calculated deflections, error vector, and RMSE.
        evaluation_cache = {}

        def evaluate_cached(E):

            E = np.asarray(E, dtype=float)

            cache_key = tuple(
                np.round(E, CACHE_DECIMAL_PLACES)
            )

            if cache_key in evaluation_cache:
                cached = evaluation_cache[cache_key]
                return (
                    cached[0].copy(),
                    cached[1].copy(),
                    cached[2]
                )

            calculated = calculate_deflections(
                thickness,
                E
            )

            if not np.all(np.isfinite(calculated)):
                error = np.ones(len(target)) * 1e6
                rmse = float(1e6)
            else:
                error = calculated - target
                rmse = float(
                    np.sqrt(np.mean(error ** 2))
                )

            evaluation_cache[cache_key] = (
                calculated.copy(),
                error.copy(),
                rmse
            )

            return calculated, error, rmse


        # ========================================================
        # RESIDUAL FUNCTION
        # ========================================================

        def residual(E):


            nonlocal evaluation_number, last_history_E


            calculated, error, rmse = evaluate_cached(E)


            # ----------------------------------------------------
            # INVALID RESULT            # ----------------------------------------------------
            # STORE HISTORY WITHOUT DUPLICATE MODULUS VECTORS
            # ----------------------------------------------------
            # least_squares can evaluate the same E several times.
            # Do not create a new Excel row unless the modulus vector
            # has materially changed from the previous recorded row
            # in this optimization stage.
            if (
                last_history_E is None
                or not np.allclose(
                    E,
                    last_history_E,
                    rtol=HISTORY_DEDUP_RTOL,
                    atol=HISTORY_DEDUP_ATOL
                )
            ):

                evaluation_number += 1

                hit_lower_live, hit_upper_live = check_bound_hit(
                    E, lower_bounds, upper_bounds, BOUND_TOLERANCE
                )
                hit_layers_live = [
                    f"E{i + 1} {'lower' if hit_lower_live[i] else 'upper'}"
                    for i in range(N)
                    if hit_lower_live[i] or hit_upper_live[i]
                ]
                history_row = {
                    "Stage": stage,
                    "Evaluation": evaluation_number,
                    "RMSE": rmse,
                    "Boundary_Hit": bool(hit_layers_live),
                    "Boundary_Hit_Layers": ", ".join(hit_layers_live),
                }


                for i in range(N):


                    history_row[
                        f"E{i + 1}"
                    ] = float(
                        E[i]
                    )


                    history_row[
                        f"Lower_Bound_E{i + 1}"
                    ] = float(
                        lower_bounds[i]
                    )


                    history_row[
                        f"Upper_Bound_E{i + 1}"
                    ] = float(
                        upper_bounds[i]
                    )


                history.append(
                    history_row
                )

                last_history_E = np.asarray(
                    E,
                    dtype=float
                ).copy()

                # Update Streamlit's live plots during the actual optimizer run.
                # The callback is optional, so the numerical solver remains usable
                # outside the Streamlit interface as well.
                if LIVE_PROGRESS_CALLBACK is not None:
                    try:
                        LIVE_PROGRESS_CALLBACK(
                            history=list(history),
                            E=np.asarray(E, dtype=float).copy(),
                            calculated=np.asarray(calculated, dtype=float).copy(),
                            target=np.asarray(target, dtype=float).copy(),
                            offsets=np.asarray(OFFSETS, dtype=float).copy(),
                            lower_bounds=np.asarray(lower_bounds, dtype=float).copy(),
                            upper_bounds=np.asarray(upper_bounds, dtype=float).copy(),
                            stage=stage,
                        )
                    except Exception as live_exc:
                        # A display issue must not interrupt the back-calculation.
                        print(f"Live graph update warning: {live_exc}")

            # Stop at the first evaluated solution meeting the user's tolerance.
            # Otherwise least_squares keeps minimizing toward ~1e-12 mm even when
            # the requested tolerance is much larger (for example, 0.0001 mm).
            if rmse <= target_rmse:
                raise _TargetRMSEReached(E, calculated, error, rmse)

            return error


        # ========================================================
        # STARTING POINTS
        # ========================================================

        starts = get_start_points(

            lower_bounds,

            upper_bounds,

            global_best_E,

            stage

        )


        # ========================================================
        # MULTIPLE START OPTIMIZATION
        # ========================================================

        for start_number, x0 in enumerate(

            starts,
            start=1

        ):


            print()

            print(

                f"Running start "

                f"{start_number}"

                f"/"

                f"{len(starts)}"

            )


            try:

                # Stop least_squares immediately when residual() detects that the
                # user-entered RMSE tolerance has been met.
                try:
                    optimizer_result = least_squares(
                        residual,
                        x0=x0,
                        bounds=(lower_bounds, upper_bounds),
                        method="trf",
                        jac="2-point",
                        x_scale="jac",
                        loss="linear",
                        ftol=1e-10,
                        xtol=1e-10,
                        gtol=1e-10,
                        max_nfev=MAX_NFEV,
                        verbose=0,
                    )
                    candidate_E = np.clip(optimizer_result.x, lower_bounds, upper_bounds)
                except _TargetRMSEReached as reached:
                    optimizer_result = None
                    candidate_E = np.clip(reached.E, lower_bounds, upper_bounds)


                # ------------------------------------------------
                # FORWARD CALCULATION
                # ------------------------------------------------

                candidate_calculated, candidate_error, candidate_rmse = (
                    evaluate_cached(candidate_E)
                )


                print(

                    f"Start {start_number} RMSE: "

                    f"{candidate_rmse:.10e}"

                )


                # ------------------------------------------------
                # STAGE BEST
                # ------------------------------------------------

                if (

                    candidate_rmse
                    <
                    stage_best_rmse

                ):


                    stage_best_rmse = (

                        candidate_rmse

                    )


                    stage_best_E = (

                        candidate_E.copy()

                    )


                    stage_best_calculated = (

                        candidate_calculated.copy()

                    )


                    stage_best_optimizer = (

                        optimizer_result

                    )


                # ------------------------------------------------
                # GLOBAL BEST
                # ------------------------------------------------

                if (

                    candidate_rmse
                    <
                    global_best_rmse

                ):


                    global_best_rmse = (

                        candidate_rmse

                    )


                    global_best_E = (

                        candidate_E.copy()

                    )


                    global_best_calculated = (

                        candidate_calculated.copy()

                    )


                    global_best_optimizer = (

                        optimizer_result

                    )


                    global_best_lower_bounds = (

                        lower_bounds.copy()

                    )


                    global_best_upper_bounds = (

                        upper_bounds.copy()

                    )


                # ------------------------------------------------
                # TARGET REACHED
                # ------------------------------------------------

                if (

                    candidate_rmse
                    <=
                    target_rmse

                ):

                    break


            except Exception as exc:


                print()

                print(

                    "Optimization error:",

                    exc

                )


                continue


        # ========================================================
        # CHECK STAGE RESULT
        # ========================================================

        if stage_best_E is None:


            print()

            print(

                "No valid solution found in this stage."

            )


            break


        print()

        print("-" * 80)

        print(

            f"BEST RESULT AT STAGE {stage}"

        )

        print("-" * 80)


        for i in range(N):


            print(

                f"E{i + 1} = "

                f"{stage_best_E[i]:.6f} MPa"

            )


        print()

        print(

            f"RMSE = "

            f"{stage_best_rmse:.10e} mm"

        )


        # ========================================================
        # TARGET ACHIEVED
        # ========================================================

        if (

            stage_best_rmse
            <=
            target_rmse

        ):


            print()

            print(

                "TARGET RMSE ACHIEVED."

            )


            break


        # ========================================================
        # CHECK BOUND HITS
        # ========================================================

        hit_lower, hit_upper = (

            check_bound_hit(

                stage_best_E,

                lower_bounds,

                upper_bounds,

                BOUND_TOLERANCE

            )

        )


        print()

        print(

            "BOUNDARY CHECK"

        )


        any_bound_hit = False


        for i in range(N):


            if hit_lower[i]:


                print(

                    f"E{i + 1} is near the LOWER bound."

                )


                any_bound_hit = True


            if hit_upper[i]:


                print(

                    f"E{i + 1} is near the UPPER bound."

                )


                any_bound_hit = True


        # ========================================================
        # NO BOUND HIT
        # ========================================================

        if not any_bound_hit:


            print()

            print(

                "RMSE target was not achieved, but the solution "
                "is not hitting any modulus boundary."

            )


            print(

                "Therefore, automatic expansion is stopped."

            )


            break


        # ========================================================
        # CREATE NEXT-STAGE BOUNDS
        # ========================================================
        #
        # KEY CHANGE:
        #
        # 1. The modulus that hit a bound is expanded.
        #
        # 2. Every modulus that did NOT hit a bound is kept near
        #    its previous recovered value.
        #
        # Example:
        #
        #     E1 hits upper bound -> E1 upper + 1000
        #     E2 = 70             -> E2 = 60 to 80
        #     E3 = 20             -> E3 = 10 to 30
        #
        # This allows the forward solver to re-adjust E2 and E3,
        # but prevents them from jumping over the entire original
        # search range.
        # ========================================================

        new_lower_bounds, new_upper_bounds = (

            create_next_stage_bounds(

                lower_bounds,

                upper_bounds,

                stage_best_E,

                hit_lower,

                hit_upper

            )

        )


        # ========================================================
        # CHECK WHETHER ANY BOUND CHANGED
        # ========================================================

        bounds_changed = (

            not np.allclose(

                new_lower_bounds,

                lower_bounds

            )

            or

            not np.allclose(

                new_upper_bounds,

                upper_bounds

            )

        )


        if not bounds_changed:

            print()

            print(

                "Bounds cannot be expanded further."

            )

            break


        # ========================================================
        # DISPLAY NEXT-STAGE BOUNDS
        # ========================================================

        print()

        print(

            "CREATING NEXT OPTIMIZATION SEARCH REGION..."

        )


        for i in range(N):

            if hit_lower[i]:

                print(

                    f"E{i + 1}: LOWER bound hit -> "

                    f"expanded to "

                    f"{new_lower_bounds[i]:.4f} to "

                    f"{new_upper_bounds[i]:.4f} MPa"

                )

            elif hit_upper[i]:

                print(

                    f"E{i + 1}: UPPER bound hit -> "

                    f"expanded to "

                    f"{new_lower_bounds[i]:.4f} to "

                    f"{new_upper_bounds[i]:.4f} MPa"

                )

            else:

                print(

                    f"E{i + 1}: preserved around "

                    f"{stage_best_E[i]:.4f} MPa -> "

                    f"{new_lower_bounds[i]:.4f} to "

                    f"{new_upper_bounds[i]:.4f} MPa"

                )


        # ========================================================
        # UPDATE BOUNDS
        # ========================================================

        lower_bounds = new_lower_bounds

        upper_bounds = new_upper_bounds


    # ============================================================
    # FINAL CHECK
    # ============================================================

    if global_best_E is None:


        return {

            "success":
                False,

            "history":
                history

        }


    # ============================================================
    # FINAL RESULTS
    # ============================================================

    final_E = (

        global_best_E.copy()

    )


    final_calculated = (

        calculate_deflections(

            thickness,

            final_E

        )

    )


    final_error = (

        final_calculated
        -
        target

    )


    absolute_error = np.abs(

        final_error

    )


    rmse = float(

        np.sqrt(

            np.mean(

                final_error ** 2

            )

        )

    )


    mae = float(

        np.mean(

            absolute_error

        )

    )


    maximum_error = float(

        np.max(

            absolute_error

        )

    )


    percentage_error = np.where(

        np.abs(target) > 1e-15,

        absolute_error
        /
        np.abs(target)
        *
        100.0,

        0.0

    )


    r2 = calculate_r2(

        target,

        final_calculated

    )


    return {


        "success":
            True,


        "Back_E":
            final_E,


        "Calculated":
            final_calculated,


        "Target":
            target,


        "Error":
            final_error,


        "Percentage_Error":
            percentage_error,


        "RMSE":
            rmse,


        "MAE":
            mae,


        "Maximum_Error":
            maximum_error,


        "R2":
            r2,


        "Target_Achieved":

            rmse <= target_rmse,


        "Best_Lower_Bounds":

            global_best_lower_bounds,


        "Best_Upper_Bounds":

            global_best_upper_bounds,


        "NFEV":

            (

                global_best_optimizer.nfev

                if global_best_optimizer is not None

                else np.nan

            ),


        "History":
            history

    }


# ================================================================
# 12. CREATE RESULTS DATAFRAME
# ================================================================

def create_results_dataframe(

    result

):


    row = {}


    # ============================================================
    # RECOVERED MODULI
    # ============================================================

    for i in range(N):


        row[

            f"Back E{i + 1} (MPa)"

        ] = result["Back_E"][i]


    # ============================================================
    # BOUNDS
    # ============================================================

    for i in range(N):


        row[

            f"Initial Lower E{i + 1}"

        ] = INITIAL_LOWER_BOUNDS[i]


        row[

            f"Initial Upper E{i + 1}"

        ] = INITIAL_UPPER_BOUNDS[i]


        row[

            f"Best Lower Bound E{i + 1}"

        ] = result["Best_Lower_Bounds"][i]


        row[

            f"Best Upper Bound E{i + 1}"

        ] = result["Best_Upper_Bounds"][i]


    # ============================================================
    # ERROR STATISTICS
    # ============================================================

    row["RMSE (mm)"] = result["RMSE"]

    row["MAE (mm)"] = result["MAE"]

    row["Maximum Error (mm)"] = (

        result["Maximum_Error"]

    )

    row["R2"] = result["R2"]

    row["Target Achieved"] = (

        result["Target_Achieved"]

    )

    row["NFEV"] = result["NFEV"]


    # ============================================================
    # DEFLECTION RESULTS
    # ============================================================

    for i in range(NUMBER_OF_POINTS):


        radius = OFFSETS[i]


        row[

            f"Target Point {i + 1}"

        ] = result["Target"][i]


        row[

            f"Calculated Point {i + 1}"

        ] = result["Calculated"][i]


        row[

            f"Error Point {i + 1}"

        ] = result["Error"][i]


        row[

            f"Percentage Error Point {i + 1}"

        ] = result["Percentage_Error"][i]


        row[

            f"Radius Point {i + 1} (mm)"

        ] = radius


    return pd.DataFrame(

        [row]

    )


# ================================================================
# 13. CREATE DEFLECTION GRAPH
# ================================================================

def create_deflection_graph(

    result

):


    plt.figure(

        figsize=(9, 6)

    )


    plt.plot(

        OFFSETS,

        result["Target"],

        marker="o",

        linewidth=2,

        label="Target Deflection"

    )


    plt.plot(

        OFFSETS,

        result["Calculated"],

        marker="s",

        linestyle="--",

        linewidth=2,

        label="Back-calculated Deflection"

    )


    plt.xlabel(

        "Radial Distance (mm)"

    )


    plt.ylabel(

        "Deflection (mm)"

    )


    plt.title(

        "Target vs Back-calculated Deflections"

    )


    plt.grid(

        True,

        alpha=0.3

    )


    plt.legend()


    plt.tight_layout()


    graph_file = os.path.join(

        OUTPUT_FOLDER,

        "Deflection_Comparison.png"

    )


    plt.savefig(

        graph_file,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


# ================================================================
# 14. CREATE RMSE CONVERGENCE GRAPH
# ================================================================

def create_rmse_graph(

    history

):


    if len(history) == 0:

        return


    history_df = pd.DataFrame(

        history

    )


    if "RMSE" not in history_df.columns:

        return


    plt.figure(

        figsize=(9, 6)

    )


    plt.plot(

        history_df["Evaluation"],

        history_df["RMSE"],

        linewidth=1.5

    )


    plt.xlabel(

        "Function Evaluation"

    )


    plt.ylabel(

        "RMSE (mm)"

    )


    plt.title(

        "RMSE Convergence"

    )


    plt.yscale(
        "log"
    )


    plt.grid(

        True,

        alpha=0.3

    )


    plt.tight_layout()


    graph_file = os.path.join(

        OUTPUT_FOLDER,

        "RMSE_Convergence.png"

    )


    plt.savefig(

        graph_file,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()





# ================================================================
# PAVE INVERSE - USER INTERFACE
# ================================================================
# EDIT THE WELCOME MESSAGE HERE whenever you want to change the landing-page text.
WELCOME_MESSAGE = (
    "Welcome to PAVE INVERSE - ADVANCED FWD BACK CALCULATION. "
    "Please choose the type of analysis you want to do: "
    "One-to-one analysis or Excel Input based analysis."
)

BASE_MAX_BOUND_EXPANSIONS = 100
APP_DIR = Path(__file__).resolve().parent
TEMPLATE_FILENAME = "Standard Format _ PAVE INVERSE.xlsx"
TEMPLATE_PATH = APP_DIR / TEMPLATE_FILENAME


def _reset_to_home():
    st.session_state["analysis_mode"] = None
    st.session_state.pop("one_result", None)
    st.session_state.pop("one_input_signature", None)
    st.session_state.pop("one_stage0_done", None)
    st.session_state.pop("one_expand_requested", None)


def _make_bowl_figure(offsets, target, calculated, bound_hit=False, title="Target vs Back-calculated Deflection Bowl"):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(offsets, target, marker="o", linewidth=2, label="Target deflection")
    ax.plot(offsets, calculated, marker="s", linestyle="--", linewidth=2, label="Back-calculated deflection")
    ax.set_xlabel("Radial distance from load centre (mm)")
    ax.set_ylabel("Deflection (mm)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    if bound_hit:
        ax.text(
            0.5, 0.93, "MODULUS BOUND HIT!",
            transform=ax.transAxes, ha="center", va="top",
            color="red", fontsize=14, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="red", alpha=0.95, boxstyle="round,pad=0.4")
        )
        ax.spines["top"].set_color("red")
        ax.spines["top"].set_linewidth(2.5)
        ax.spines["right"].set_color("red")
        ax.spines["right"].set_linewidth(2.5)
    fig.tight_layout()
    return fig


def _configure_solver(thickness, poisson, lower, upper, pressure, radius, offsets, targets, target_rmse):
    global N, THICKNESS, INITIAL_LOWER_BOUNDS, INITIAL_UPPER_BOUNDS, POISSON
    global PRESSURE_MPA, LOAD_RADIUS_MM, NUMBER_OF_POINTS, OFFSETS, POINTS, TARGET
    global TARGET_RMSE, OUTPUT_FOLDER
    N = len(lower)
    THICKNESS = np.asarray(thickness, dtype=float)
    INITIAL_LOWER_BOUNDS = np.asarray(lower, dtype=float)
    INITIAL_UPPER_BOUNDS = np.asarray(upper, dtype=float)
    POISSON = np.asarray(poisson, dtype=float)
    PRESSURE_MPA = float(pressure)
    LOAD_RADIUS_MM = float(radius)
    NUMBER_OF_POINTS = len(offsets)
    OFFSETS = np.asarray(offsets, dtype=float)
    POINTS = [{"r": float(r), "z": 0.0, "layer": 1} for r in OFFSETS]
    TARGET = np.asarray(targets, dtype=float)
    TARGET_RMSE = float(target_rmse)
    OUTPUT_FOLDER = tempfile.mkdtemp(prefix="pave_inverse_")


def _bound_hit_for_result(result):
    if not result or not result.get("success"):
        return False, [], []
    low, high = check_bound_hit(
        result["Back_E"], result["Best_Lower_Bounds"],
        result["Best_Upper_Bounds"], BOUND_TOLERANCE
    )
    names = []
    for i, (lo, hi) in enumerate(zip(low, high)):
        if lo:
            names.append(f"E{i+1} lower")
        if hi:
            names.append(f"E{i+1} upper")
    return bool(names), names, (low, high)


def _show_one_to_one_page():
    global MAX_BOUND_EXPANSIONS, LIVE_PROGRESS_CALLBACK
    st.title("PAVE INVERSE — One-to-one analysis")
    if st.button("← Back to analysis selection"):
        _reset_to_home()
        st.rerun()

    # Clean starting form: only the structural counts are pre-populated.
    # All engineering input fields start at zero for a fresh one-to-one analysis.
    n = int(st.number_input("Number of layers", min_value=2, max_value=10, value=2, step=1, key="one_n"))
    point_count = int(st.number_input("Number of deflection points", min_value=1, max_value=16, value=1, step=1, key="one_point_count"))

    with st.form("one_to_one_inputs"):
        st.subheader("Layer properties")
        thickness, poisson, lower, upper = [], [], [], []
        for i in range(n):
            st.markdown(f"**Layer {i+1}**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if i < n - 1:
                    thickness.append(st.number_input(f"Thickness (mm), Layer {i+1}", min_value=0.0, value=0.0, step=10.0, key=f"one_th_{i}"))
                else:
                    st.write("Semi-infinite layer")
            with c2:
                poisson.append(st.number_input(f"Poisson ratio, Layer {i+1}", min_value=0.0, max_value=0.499, value=0.0, step=0.01, format="%.3f", key=f"one_nu_{i}"))
            with c3:
                lower.append(st.number_input(f"E{i+1} minimum (MPa)", min_value=0.0, value=0.0, step=10.0, key=f"one_lo_{i}"))
            with c4:
                upper.append(st.number_input(f"E{i+1} maximum (MPa)", min_value=0.0, value=0.0, step=10.0, key=f"one_hi_{i}"))

        st.subheader("Loading")
        c1, c2, c3 = st.columns(3)
        with c1:
            pressure = st.number_input("Loading pressure (MPa)", min_value=0.0, value=0.0, format="%.6f", key="one_pressure")
        with c2:
            radius = st.number_input("Load radius (mm)", min_value=0.0, value=0.0, step=5.0, key="one_radius")
        with c3:
            target_rmse = st.number_input("Target RMSE (mm)", min_value=0.0, value=0.0, format="%.12g", key="one_rmse")

        st.subheader("Target deflections")
        st.caption("Depth is fixed at 0 mm and layer number is fixed at 1 for every evaluation point.")
        offsets, targets = [], []
        for i in range(point_count):
            a, b = st.columns(2)
            with a:
                offsets.append(st.number_input(f"Radial distance (mm), point {i+1}", min_value=0.0, value=0.0, step=50.0, key=f"one_r_{i}"))
            with b:
                targets.append(st.number_input(f"Target deflection (mm), point {i+1}", min_value=0.0, value=0.0, format="%.12g", key=f"one_target_{i}"))
        submitted = st.form_submit_button("Run Stage 0", type="primary", use_container_width=True)

    if submitted:
        # Zero is used as the blank/default display value, but these quantities
        # must be populated with physically valid positive values before solving.
        if any(value <= 0 for value in thickness):
            st.error("Enter a thickness greater than 0 mm for every finite layer.")
            return
        if any(value <= 0 or value >= 0.5 for value in poisson):
            st.error("Enter a Poisson ratio greater than 0 and less than 0.5 for every layer.")
            return
        if any(value <= 0 for value in lower) or any(value <= 0 for value in upper):
            st.error("Enter positive minimum and maximum modulus values for every layer.")
            return
        if pressure <= 0:
            st.error("Enter a loading pressure greater than 0 MPa.")
            return
        if radius <= 0:
            st.error("Enter a load radius greater than 0 mm.")
            return
        if target_rmse <= 0:
            st.error("Enter a target RMSE greater than 0 mm.")
            return
        if any(upper[i] <= lower[i] for i in range(n)):
            st.error("Each maximum modulus must be greater than its minimum modulus.")
            return
        if len(set(offsets)) != len(offsets):
            st.error("Radial distances must be unique.")
            return
        _configure_solver(thickness, poisson, lower, upper, pressure, radius, offsets, targets, target_rmse)
        st.session_state["one_input_signature"] = (n, tuple(thickness), tuple(poisson), tuple(lower), tuple(upper), pressure, radius, tuple(offsets), tuple(targets), target_rmse)
        st.session_state["one_expand_requested"] = False
        st.session_state["one_stage0_done"] = False
        st.session_state.pop("one_result", None)
        st.session_state["one_run_requested"] = True
        st.rerun()

    # Execute Stage 0 once; subsequent reruns show the retained result and the user decision.
    if st.session_state.get("one_run_requested"):
        sig = st.session_state.get("one_input_signature")
        if sig:
            n0, th0, nu0, lo0, hi0, p0, rad0, off0, tgt0, trmse0 = sig
            _configure_solver(th0, nu0, lo0, hi0, p0, rad0, off0, tgt0, trmse0)
            MAX_BOUND_EXPANSIONS = BASE_MAX_BOUND_EXPANSIONS if st.session_state.get("one_expand_requested") else 0
            progress = st.progress(0, text="Running adaptive multi-start least-squares optimisation…")
            live_status = st.empty()
            live_moduli = st.empty()
            plot_slot = st.empty()
            log = io.StringIO()
            last_eval = {"value": 0}

            def live_bowl(history, E, calculated, target, offsets, lower_bounds, upper_bounds, stage):
                if not len(history):
                    return
                row = history[-1]
                hit = bool(row.get("Boundary_Hit", False))
                if len(history) > 1 and len(history) % 3 != 0 and not hit:
                    return
                live_status.info(f"Live optimisation · Stage {stage} · evaluation {row.get('Evaluation', len(history))} · RMSE {row.get('RMSE', np.nan):.6e} mm")
                live_moduli.markdown("**Current moduli (MPa):** " + "  |  ".join(f"E{i+1} = {v:.5f}" for i, v in enumerate(E)))
                fig = _make_bowl_figure(offsets, target, calculated, hit, "Live deflection bowl")
                plot_slot.pyplot(fig, use_container_width=True)
                plt.close(fig)

            LIVE_PROGRESS_CALLBACK = live_bowl
            try:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    result = adaptive_backward_solver(THICKNESS, TARGET, INITIAL_LOWER_BOUNDS, INITIAL_UPPER_BOUNDS)
                st.session_state["one_result"] = result
                st.session_state["one_log"] = log.getvalue()
                st.session_state["one_stage0_done"] = True
                st.session_state["one_completed_expanded"] = bool(st.session_state.get("one_expand_requested"))
                st.session_state["one_run_requested"] = False
                progress.progress(100, text="Calculation finished.")
            except Exception as exc:
                st.error(f"Calculation failed: {exc}")
                with st.expander("Diagnostic log"):
                    st.code(log.getvalue() or repr(exc))
                st.session_state["one_run_requested"] = False
                return
            finally:
                LIVE_PROGRESS_CALLBACK = None
            st.rerun()

    result = st.session_state.get("one_result")
    if not result:
        st.info("Enter the inputs above and select **Run Stage 0**. The solver will stop after Stage 0 so you can decide whether to expand bounds.")
        return

    # Streamlit reruns the script from the top; restore solver globals from the
    # saved input signature before rendering retained results or download files.
    saved_signature = st.session_state.get("one_input_signature")
    if saved_signature:
        n0, th0, nu0, lo0, hi0, p0, rad0, off0, tgt0, trmse0 = saved_signature
        _configure_solver(th0, nu0, lo0, hi0, p0, rad0, off0, tgt0, trmse0)
    else:
        trmse0 = float(st.session_state.get("one_target_rmse", 0.0001))

    if not result.get("success"):
        st.error("No valid solution was found.")
        with st.expander("Solver log"):
            st.code(st.session_state.get("one_log", ""))
        return

    bound_hit, hit_names, _ = _bound_hit_for_result(result)
    achieved = result["RMSE"] <= float(trmse0)
    if achieved:
        st.success(f"TARGET RMSE REACHED — RMSE = {result['RMSE']:.6e} mm")
    elif bound_hit:
        st.error("MODULUS BOUNDS HIT — " + ", ".join(hit_names))
        if not st.session_state.get("one_completed_expanded"):
            st.warning("Stage 0 has ended. Do you want to expand the modulus bounds and continue with Stage 1 and subsequent stages, or end the calculation here?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Expand bounds and continue", type="primary", use_container_width=True):
                    st.session_state["one_expand_requested"] = True
                    st.session_state["one_run_requested"] = True
                    st.rerun()
            with c2:
                if st.button("End calculation here", use_container_width=True):
                    st.session_state["one_completed_expanded"] = True
                    st.session_state["one_expand_requested"] = False
                    st.rerun()
    else:
        st.warning("Stage 0 ended without reaching the target RMSE, and no modulus bound was hit. The solver has stopped.")

    st.subheader("Final / current results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE (mm)", f"{result['RMSE']:.6e}")
    c2.metric("MAE (mm)", f"{result['MAE']:.6e}")
    c3.metric("Maximum error (mm)", f"{result['Maximum_Error']:.6e}")
    c4.metric("R²", f"{result['R2']:.8f}" if np.isfinite(result["R2"]) else "N/A")

    modulus_df = pd.DataFrame({
        "Layer": [f"E{i+1}" for i in range(len(result["Back_E"]))],
        "Recovered modulus (MPa)": result["Back_E"],
        "Initial minimum (MPa)": INITIAL_LOWER_BOUNDS,
        "Initial maximum (MPa)": INITIAL_UPPER_BOUNDS,
        "Best-stage lower bound (MPa)": result["Best_Lower_Bounds"],
        "Best-stage upper bound (MPa)": result["Best_Upper_Bounds"],
    })
    st.dataframe(modulus_df, use_container_width=True, hide_index=True)
    comparison_df = pd.DataFrame({
        "Radius (mm)": OFFSETS, "Target deflection (mm)": result["Target"],
        "Back-calculated deflection (mm)": result["Calculated"], "Error (mm)": result["Error"],
        "Absolute error (mm)": np.abs(result["Error"]), "Percentage error (%)": result["Percentage_Error"]
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    final_fig = _make_bowl_figure(OFFSETS, result["Target"], result["Calculated"], bound_hit)
    st.pyplot(final_fig, use_container_width=True)
    png_buffer = io.BytesIO()
    final_fig.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight")
    plt.close(final_fig)
    st.download_button("Download final deflection bowl (PNG)", png_buffer.getvalue(), file_name="Deflection_Bowl.png", mime="image/png")

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        modulus_df.to_excel(writer, sheet_name="Recovered Moduli", index=False)
        comparison_df.to_excel(writer, sheet_name="Deflection Comparison", index=False)
        pd.DataFrame(result.get("History", [])).to_excel(writer, sheet_name="Stage History", index=False)
        pd.DataFrame({
            "Parameter": ["Number of layers", "Thicknesses (mm)", "Poisson ratios", "Pressure (MPa)", "Load radius (mm)", "Target RMSE (mm)", "Final RMSE (mm)", "Target reached", "Bound hit", "Bound-hit layers"],
            "Value": [len(result["Back_E"]), str(list(THICKNESS)), str(list(POISSON)), PRESSURE_MPA, LOAD_RADIUS_MM, TARGET_RMSE, result["RMSE"], achieved, bound_hit, ", ".join(hit_names)]
        }).to_excel(writer, sheet_name="Input Summary", index=False)
    st.download_button("Download inputs and final results (Excel)", output.getvalue(), file_name="PAVE_INVERSE_One_to_one_Results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    with st.expander("Solver log"):
        st.code(st.session_state.get("one_log", "No log available."))


def _numeric(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        v = float(value)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _process_excel_upload(uploaded_file, target_rmse):
    import zipfile
    source_bytes = uploaded_file.getvalue()
    input_wb = openpyxl.load_workbook(io.BytesIO(source_bytes))
    # Use the named input sheet from the supplied template.
    ws = input_wb["Standard Universal Data"] if "Standard Universal Data" in input_wb.sheetnames else input_wb.active
    n = _numeric(ws["C1"].value)
    if n is None or int(n) < 2 or int(n) > 10:
        raise ValueError("Cell C1 must contain a number of layers from 2 to 10.")
    n = int(n)

    thickness = [_numeric(ws.cell(row=5+i, column=3).value) for i in range(n-1)]
    poisson = [_numeric(ws.cell(row=5+i, column=5).value) for i in range(n)]
    lower = [_numeric(ws.cell(row=5+i, column=7).value) for i in range(n)]
    upper = [_numeric(ws.cell(row=5+i, column=9).value) for i in range(n)]
    if any(v is None or v <= 0 for v in thickness):
        raise ValueError("Enter a positive thickness in column C for each finite layer (rows 5 through 4+N).")
    if any(v is None or not (0 < v < 0.5) for v in poisson):
        raise ValueError("Enter a Poisson ratio between 0 and 0.5 in column E for all layers.")
    if any(lower[i] is None or upper[i] is None or lower[i] < 1 or upper[i] <= lower[i] for i in range(n)):
        raise ValueError("Enter valid minimum and maximum moduli in columns G and I for every layer.")

    offsets = []
    offset_columns = []
    for col in range(8, 17):  # H:P; includes the two extra optional sensor columns
        v = _numeric(ws.cell(row=19, column=col).value)
        if v is not None:
            offsets.append(v)
            offset_columns.append(col)
    if not offsets:
        raise ValueError("No numeric deflection-point distances were found in row 19, columns H:P.")

    from copy import copy
    plot_files = []
    row_summaries = []
    first_data_row = 20
    last_data_row = ws.max_row
    rows_to_process = []
    for row in range(first_data_row, last_data_row+1):
        vals = [ws.cell(row=row, column=c).value for c in range(2, 17)]
        if not any(v is not None and str(v).strip() != "" for v in vals):
            continue
        # Ignore rows that only contain a serial number; a row must have pressure/radius/deflection data.
        pressure = _numeric(ws.cell(row=row, column=6).value)
        radius = _numeric(ws.cell(row=row, column=7).value)
        target_vals = [_numeric(ws.cell(row=row, column=c).value) for c in offset_columns]
        if pressure is None and radius is None and all(v is None for v in target_vals):
            continue
        if pressure is None or pressure <= 0:
            ws.cell(row=row, column=17).value = "Missing/invalid pressure in column F"
            ws.cell(row=row, column=17).font = openpyxl.styles.Font(color="FF0000", bold=True)
            continue
        if radius is None or radius <= 0:
            ws.cell(row=row, column=17).value = "Missing/invalid load radius in column G"
            ws.cell(row=row, column=17).font = openpyxl.styles.Font(color="FF0000", bold=True)
            continue
        if any(v is None for v in target_vals):
            ws.cell(row=row, column=17).value = "Missing/non-numeric deflection data in H:P"
            ws.cell(row=row, column=17).font = openpyxl.styles.Font(color="FF0000", bold=True)
            continue
        # The requested image serial is the workbook's ID No. (column E), not Station (column A).
        serial = ws.cell(row=row, column=5).value
        if serial is None or str(serial).strip() == "":
            serial = ws.cell(row=row, column=1).value  # fallback only when ID No. is blank
        serial_num = str(serial).strip() if serial is not None else str(row-first_data_row+1)
        # Keep filenames safe while preserving the actual ID where possible.
        serial_num = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in serial_num).strip("_") or str(row-first_data_row+1)
        rows_to_process.append((row, serial_num, pressure, radius, target_vals))

    if not rows_to_process:
        raise ValueError("No valid FWD input rows were found. Fill each row's pressure (F), load radius (G), and deflections under the distances in row 19.")

    images = {}
    old_max = MAX_BOUND_EXPANSIONS
    old_callback = LIVE_PROGRESS_CALLBACK
    try:
        for row, serial, pressure, radius, target_vals in rows_to_process:
            _configure_solver(thickness, poisson, lower, upper, pressure, radius, offsets, target_vals, target_rmse)
            # Excel mode always stops after Stage 0; no live graphs and no automatic expansion.
            globals()["MAX_BOUND_EXPANSIONS"] = 0
            globals()["LIVE_PROGRESS_CALLBACK"] = None
            log = io.StringIO()
            try:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    result = adaptive_backward_solver(THICKNESS, TARGET, INITIAL_LOWER_BOUNDS, INITIAL_UPPER_BOUNDS, target_rmse=target_rmse)
            except Exception as exc:
                ws.cell(row=row, column=17).value = f"Calculation failed: {exc}"
                ws.cell(row=row, column=17).font = openpyxl.styles.Font(color="FF0000", bold=True)
                continue
            if not result.get("success"):
                ws.cell(row=row, column=17).value = "Calculation failed: no valid solution"
                ws.cell(row=row, column=17).font = openpyxl.styles.Font(color="FF0000", bold=True)
                continue

            hit, hit_names, _ = _bound_hit_for_result(result)
            achieved = result["RMSE"] <= target_rmse
            if achieved:
                remark = (
                    "TARGET RMSE Reached (modulus bound touched)"
                    if hit else "TARGET RMSE Reached"
                )
                font = openpyxl.styles.Font(color="008000", bold=True)
            elif hit:
                remark = "Bound's Hitted, please change the bounds"
                font = openpyxl.styles.Font(color="FF0000", bold=True)
            else:
                remark = "Target RMSE not reached; no bound hit"
                font = openpyxl.styles.Font(color="FF9900", bold=True)
            ws.cell(row=row, column=17).value = remark
            ws.cell(row=row, column=17).font = font

            # Add E1...En results in columns R onwards (same output sheet).
            for i, modulus in enumerate(result["Back_E"]):
                cell = ws.cell(row=19, column=18+i)
                cell.value = f"E{i+1}"
                cell.font = copy(ws["Q19"].font)
                cell.fill = copy(ws["Q19"].fill)
                cell.alignment = copy(ws["Q19"].alignment)
                out_cell = ws.cell(row=row, column=18+i)
                out_cell.value = float(modulus)
                out_cell.number_format = "0.000000"
                out_cell.font = openpyxl.styles.Font(color="008000" if achieved else "000000")

            fig = _make_bowl_figure(offsets, result["Target"], result["Calculated"], hit, f"Station {serial} — Final deflection bowl")
            png = io.BytesIO()
            fig.savefig(png, format="png", dpi=250, bbox_inches="tight")
            plt.close(fig)
            image_name = f"{serial_num}.png"
            images[image_name] = png.getvalue()
            row_summaries.append({
                "ID No.": serial_num, "Excel row": row, "Remarks": remark,
                "RMSE (mm)": result["RMSE"], "Target RMSE reached": achieved,
                "Bound hit": hit, "Bound-hit moduli": ", ".join(hit_names),
                **{f"E{i+1} (MPa)": float(v) for i, v in enumerate(result["Back_E"])}
            })
    finally:
        globals()["MAX_BOUND_EXPANSIONS"] = old_max
        globals()["LIVE_PROGRESS_CALLBACK"] = old_callback

    # Keep the original workbook layout and only append/modify remarks and recovered modulus columns.
    output_excel = io.BytesIO()
    input_wb.save(output_excel)
    excel_bytes = output_excel.getvalue()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("PAVE_INVERSE_Excel_Results.xlsx", excel_bytes)
        for name, content in images.items():
            zf.writestr(name, content)
    return excel_bytes, zip_buffer.getvalue(), pd.DataFrame(row_summaries), len(rows_to_process), len(images)


def _show_excel_page():
    st.title("PAVE INVERSE — Excel input based analysis")
    if st.button("← Back to analysis selection", key="excel_back"):
        _reset_to_home()
        st.rerun()

    st.markdown(
        "Download the standard PAVE INVERSE workbook, enter your project and FWD data, "
        "save the completed workbook, and upload it here for analysis."
    )

    # The workbook is stored in the same GitHub repository folder as this Python file.
    # Streamlit Cloud checks out the repository, so this relative path works in deployment.
    if TEMPLATE_PATH.is_file():
        try:
            template_bytes = TEMPLATE_PATH.read_bytes()
            st.download_button(
                "Download Standard Format — PAVE INVERSE.xlsx",
                data=template_bytes,
                file_name=TEMPLATE_FILENAME,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
                key="download_pave_inverse_template",
            )
        except OSError as exc:
            st.error(f"Unable to read the standard Excel template: {exc}")
    else:
        st.error(
            f"Standard Excel template not found: `{TEMPLATE_FILENAME}`. "
            "Add this exact file to the GitHub main branch in the same folder "
            "as the Streamlit Python file, then redeploy/restart the app."
        )

    st.markdown("""
**How to fill the standard workbook**
- **C1:** number of layers (2–10).
- **Rows 5–14:** finite-layer thickness in column **C**; Poisson ratio in **E**; minimum modulus in **G**; maximum modulus in **I**.
- **Row 19:** sensor distances in **H:P**. Enter each station/drop on a separate row starting at **row 20**.
- For each station row: pressure in **F**, load radius in **G**, and target deflections under the corresponding distance columns **H:P**.
- The output workbook keeps the original layout, writes the status in **Q**, and appends recovered moduli under **R onward**. Deflection-bowl images use **ID No. in column E** as the filename (station number in column A is the fallback).
""")

    target_rmse = st.number_input(
        "Target RMSE tolerance (mm) for all rows",
        min_value=1e-12,
        value=0.0001,
        format="%.12g",
        help="Each row stops when its calculated RMSE is less than or equal to this value. The reported final RMSE can be smaller than the threshold.",
        key="excel_target_rmse",
    )
    uploaded = st.file_uploader(
        "Upload your completed standard Excel workbook (.xlsx)",
        type=["xlsx"],
        key="excel_upload",
        help="Upload the completed copy of the Standard Format — PAVE INVERSE workbook.",
    )
    if uploaded is None:
        return

    # Remove old results whenever the uploaded workbook or requested RMSE changes.
    upload_signature = hashlib.sha256(
        uploaded.getvalue() + f"|{target_rmse:.16g}".encode("utf-8")
    ).hexdigest()
    if st.session_state.get("excel_last_upload_signature") != upload_signature:
        for key in (
            "excel_output_bytes",
            "excel_zip_bytes",
            "excel_summary_df",
            "excel_row_count",
            "excel_image_count",
        ):
            st.session_state.pop(key, None)
        st.session_state["excel_last_upload_signature"] = upload_signature

    if st.button("Run Excel Stage 0 analysis", type="primary", use_container_width=True):
        with st.spinner(
            "Running Stage 0 separately for each valid station row and generating bowl images…"
        ):
            try:
                # Rewind the uploaded file before parsing, for reliable reruns.
                uploaded.seek(0)
                excel_bytes, zip_bytes, summary_df, n_rows, n_images = _process_excel_upload(
                    uploaded, target_rmse
                )
                st.session_state["excel_output_bytes"] = excel_bytes
                st.session_state["excel_zip_bytes"] = zip_bytes
                st.session_state["excel_summary_df"] = summary_df
                st.session_state["excel_row_count"] = n_rows
                st.session_state["excel_image_count"] = n_images
            except Exception as exc:
                st.error(f"Excel analysis could not be completed: {exc}")
                return

    if "excel_output_bytes" in st.session_state:
        st.success(
            f"Completed Stage 0 for {st.session_state['excel_row_count']} valid station row(s); "
            f"generated {st.session_state['excel_image_count']} deflection-bowl image(s)."
        )
        st.subheader("Processing summary")
        st.dataframe(
            st.session_state["excel_summary_df"],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download updated Excel workbook",
            data=st.session_state["excel_output_bytes"],
            file_name="PAVE_INVERSE_Excel_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
        st.download_button(
            "Download ZIP (updated workbook + serial-number PNG bowls)",
            data=st.session_state["excel_zip_bytes"],
            file_name="PAVE_INVERSE_Results_and_Deflection_Bowls.zip",
            mime="application/zip",
        )


def run_streamlit_app():
    st.set_page_config(page_title="PAVE INVERSE - Advanced FWD Back Calculation", page_icon="🛣️", layout="wide")
    if "analysis_mode" not in st.session_state:
        st.session_state["analysis_mode"] = None

    if st.session_state["analysis_mode"] is None:
        st.title("PAVE INVERSE")
        st.markdown(f"### {WELCOME_MESSAGE}")
        st.write("Select an analysis mode to continue:")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("#### One-to-one analysis")
            st.caption("Enter one FWD dataset, watch the live deflection bowl, and decide whether to expand bounds after Stage 0.")
            if st.button("One-to-one analysis", type="primary", use_container_width=True):
                st.session_state["analysis_mode"] = "one"
                st.rerun()
        with c2:
            st.markdown("#### Excel input based analysis")
            st.caption("Download the workbook template, enter multiple FWD rows, upload it, and process each row independently at Stage 0.")
            if st.button("Excel input based analysis", use_container_width=True):
                st.session_state["analysis_mode"] = "excel"
                st.rerun()
        return

    if st.session_state["analysis_mode"] == "one":
        _show_one_to_one_page()
    elif st.session_state["analysis_mode"] == "excel":
        _show_excel_page()


if __name__ == "__main__":
    run_streamlit_app()
