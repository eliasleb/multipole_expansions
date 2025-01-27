import numpy as np
import matplotlib.pyplot as plt
# import meep as mp
import pickle

import pandas as pd
from scipy.optimize import minimize
from from_mathematica import SPHERICAL_TO_CARTESIAN
from scipy.interpolate import griddata


def import_comsol(dx: int, dy: int, dz: int, r_tol: float = 1e-4, case=None, n_skip=9, compensate_symmetry=False):
    # filename = f"../../git_ignore/moments_inverse/data/{case}/{dx}{dy}{dz}.txt"
    filename = f"data/{case}/{dx}{dy}{dz}.txt"
    # with open(filename, "r") as fd:
    #     while line := fd.readline():
    #         if "real(emw.Ex)" in line:
    #             break
    # for l in line.split("@"):
    #     print(l)
    data = np.genfromtxt(
        filename,
        skip_header=n_skip
    )

    if compensate_symmetry:
        assert np.all(data[:3, 0] >= 0)  # assuming positive quadrant is given
        for xi, yi, zi, rex, iex, rey, iey, rez, iez in data:
            data = np.vstack((data, (-xi, yi, zi, rex, iex, -rey, -iey, -rez, -iez)))
        for xi, yi, zi, rex, iex, rey, iey, rez, iez in data:
            data = np.vstack((data, (xi, -yi, zi, rex, iex, -rey, -iey, rez, iez)))
        for xi, yi, zi, rex, iex, rey, iey, rez, iez in data:
            data = np.vstack((data, (xi, yi, -zi, rex, iex, rey, iey, -rez, -iez)))

    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    expected_r = np.mean(r)
    assert np.all(
        (r > (1 - r_tol) * expected_r) & (r < (1 + r_tol) * expected_r)
    )
    ex = data[:, 3] + 1j * data[:, 4]
    ey = data[:, 5] + 1j * data[:, 6]
    ez = data[:, 7] + 1j * data[:, 8]

    theta, phi, e_r, e_theta, e_phi = get_fields_on_sphere(x, y, z, ex, ey, ez)

    return x, y, z, r, theta, phi, e_r, e_theta, e_phi


def get_fields_on_sphere(x, y, z, ex, ey, ez):
    x, y, z = y, z, x
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    theta = np.arctan2(rho, z)

    r_hat = (
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    )
    theta_hat = (
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta)
    )
    phi_hat = (
        -np.sin(phi),
        np.cos(phi),
        0 * phi
    )

    e_r = ex * r_hat[0] + ey * r_hat[1] + ez * r_hat[2]
    e_theta = ex * theta_hat[0] + ey * theta_hat[1] + ez * theta_hat[2]
    e_phi = ex * phi_hat[0] + ey * phi_hat[1] + ez * phi_hat[2]
    return theta, phi, e_r, e_theta, e_phi


def spherical_vector_to_cartesian(theta, phi, vec):
    ex = np.sin(theta) * np.cos(phi) * vec[0] \
        + np.cos(theta) * np.cos(phi) * vec[1] \
        - np.sin(phi) * vec[2]
    ey = np.sin(theta) * np.sin(phi) * vec[0] \
        + np.cos(theta) * np.sin(phi) * vec[1] \
        + np.cos(phi) * vec[2]
    ez = np.cos(theta) * vec[0] - np.sin(theta) * vec[1]

    return ex, ey, ez


def plot_all(ix, iy, iz, theta, phi, e_r, e_theta, e_phi):
    plt.figure(figsize=(15, 9))

    overall_max = max(np.max(np.abs(e_r)), np.max(np.abs(e_theta)), np.max(np.abs(e_phi)))

    for plt_ind, (component, name) in enumerate(zip((e_r, e_theta, e_phi), ("E_r", "E_theta", "E_phi"))):

        plt.subplot(2, 3, plt_ind + 1)
        plt.tricontourf(
            phi, theta, np.abs(component),
            cmap="jet",
            levels=np.linspace(0, overall_max, 21)
        )
        plt.colorbar()
        plt.title(name)
        if plt_ind % 3 == 0:
            plt.ylabel("Theta")

        plt.subplot(2, 3, plt_ind + 1 + 3)
        plt.tricontourf(
            phi, theta, np.angle(component),
            cmap="hsv",
            levels=41
        )
        plt.colorbar()
        if plt_ind % 3 == 0:
            plt.ylabel("Theta")
        plt.xlabel("Phi")

        if plt_ind == 1:
            plt.title(f"{ix} {iy} {iz}")

    plt.tight_layout()


def plot_all_cartesian_results(max_order=3, case=None):
    for ix in range(max_order + 1):
        for iy in range(max_order + 1 - ix):
            for iz in range(max_order + 1 - ix - iy):
                _, _, _, r, theta, phi, e_r, e_theta, e_phi = import_comsol(ix, iy, iz, case=case)
                overall_max = max(np.max(np.abs(e_r)), np.max(np.abs(e_theta)), np.max(np.abs(e_phi)))
                e_r, e_theta, e_phi = e_r / overall_max, e_theta / overall_max, e_phi / overall_max
                plt.figure(figsize=(15, 3))
                plot_spherical_field(theta, phi, (e_r, e_theta, e_phi), title=f"{ix}, {iy}, {iz}",
                                     save_plot_data_filename=f"figs/reference_{ix}-{iy}-{iz}")
                plt.tight_layout()
                plt.savefig(f"figs/reference_{ix}_{iy}_{iz}.pdf")


def plot_all_spherical_results(max_order=3, case=None):
    theta, phi = None, None
    for l in range(0, max_order + 1):
        for m in range(-l, l + 1):
            total_field = (0, 0, 0)
            cartesian = SPHERICAL_TO_CARTESIAN[(l, m)]
            for mx, my, mz in cartesian:
                coeff = complex(cartesian[(mx, my, mz)])
                _, _, _, r, theta, phi, _e_r, _e_theta, _e_phi = import_comsol(mx, my, mz, case=case)
                total_field = coeff * _e_r + total_field[0], \
                              coeff * _e_theta + total_field[1], \
                              coeff * _e_phi + total_field[2]
            overall_max = np.max(np.abs(total_field))
            e_r, e_theta, e_phi = total_field[0] / overall_max, total_field[1] / overall_max, \
                                  total_field[2] / overall_max
            plt.figure(figsize=(15, 3))
            plot_spherical_field(theta, phi, (e_r, e_theta, e_phi), title=f"{l}, {m}",
                                 save_plot_data_filename=f"figs/reference_{l}-{m}")
            plt.tight_layout()
            plt.savefig(f"figs/reference_{l}_{m}.pdf")
    plt.show()


def compute_ertp_norm2(er, et, ep):
    return np.sum(np.abs(er)**2) + np.sum(np.abs(et)**2) + np.sum(np.abs(ep)**2)


def run_simulation(save_name, case, d2x=False, diag=(1, 1, 1), resolution=100):
    x, y, z, r, theta, phi, e_r, e_theta, e_phi = import_comsol(0, 0, 0, case=case)
    meep_simulation(x, y, z, resolution=resolution, d2x=d2x, diag=diag, save_name=save_name)
    plt.show(block=False)


def iterate_over_cartesian(max_order):
    xyz_ind = 0
    for mx in range(max_order + 1):
        for my in range(0, max(0, max_order - mx + 1)):
            for mz in range(0, max(0, max_order - mx - my + 1)):
                yield xyz_ind, mx, my, mz, 1.
                xyz_ind += 1


def iterate_over_spherical(max_order):
    lm_ind = 0
    for l in range(max_order + 1):
        for m in range(-l, l + 1):
            yield lm_ind, l, m, 1.
            lm_ind += 1


def iterate_over_spherical_then_cartesian(max_order, return_lm=False):
    lm_ind = 0
    for l in range(0, max_order + 1):
        for m in range(-l, l + 1):
            cartesian = SPHERICAL_TO_CARTESIAN[(l, m)]
            for mx, my, mz in cartesian:
                coeff = complex(cartesian[(mx, my, mz)])
                if return_lm:
                    yield lm_ind, mx, my, mz, coeff, l, m
                else:
                    yield lm_ind, mx, my, mz, coeff
            lm_ind += 1


def sum_over_sphere(theta, field):
    assert np.all(theta >= 0) and np.all(theta <= np.pi)
    return np.nansum(field * np.sin(theta))


def norm(theta, er, et, ep, p=2):
    return sum_over_sphere(theta, np.abs(er)**p) \
           + sum_over_sphere(theta, np.abs(et)**p) \
           + sum_over_sphere(theta, np.abs(ep)**p)
    # return max(np.max(np.abs(er)), np.max(np.abs(et)), np.max(np.abs(ep)))


def phi_split_sinusoidal(phi, theta):
    assert np.all((0 <= phi)*(phi <= 2 * np.pi))
    return (phi - np.pi) * np.sin(theta) + np.pi


def plot_spherical_grid():
    theta = np.linspace(0, np.pi, 100)
    args = ("w-", )
    kwargs = dict(linewidth=.8)
    for longitude in np.arange(-180, 181, 45):
        phi = np.sin(theta) * longitude
        plt.plot(180 + phi, theta*180/np.pi, *args, **kwargs)
    phi = np.linspace(0, 360, 2)
    for latitude in np.arange(0, 181, 45):
        plt.plot(phi, (latitude, latitude), *args, **kwargs)

    plt.yticks(list(range(0, 181, 45)))


def plot_spherical_field(
        theta, phi, e, e_comp=None, title="", save_plot_data_filename=None
):
    kwargs = dict(cmap="jet", levels=np.linspace(0, 1, 11))
    to_plot = spherical_vector_to_cartesian(theta, phi, e)
    e_max = np.max(np.abs(np.array(to_plot)))
    to_plot /= e_max
    if e_comp is not None:
        comp_to_plot = spherical_vector_to_cartesian(theta, phi, e_comp)
        comp_max = np.max(np.abs(np.array(comp_to_plot)))
        comp_to_plot /= comp_max
        n_rows = 2
    else:
        comp_to_plot = None
        n_rows = 1
    # for i, (meas, fitted) in enumerate(zip(to_plot, fitted_for_plot)):
    #     to_plot[i][np.isnan(meas)] = 0.
    #     fitted_for_plot[i][np.isnan(fitted)] = 0.
    phi_for_plot = phi_split_sinusoidal(phi + np.pi, theta) * 180 / np.pi

    def set_lims():
        plt.xlim(np.min(phi_for_plot), np.max(phi_for_plot))
        plt.ylim(np.min(theta) * 180 / np.pi, np.max(theta) * 180 / np.pi)

    plt.clf()
    plt.subplot(n_rows, 3, 1)
    args = (phi_for_plot, theta * 180 / np.pi, np.abs(to_plot[0]))
    plt.tricontourf(*args, **kwargs)
    if save_plot_data_filename is not None:
        np.savetxt(
            f"{save_plot_data_filename}_x.txt",
            np.stack(args)
        )
    plt.ylabel("θ (°)")
    plot_spherical_grid()
    set_lims()

    plt.subplot(n_rows, 3, 2)
    plt.tricontourf(phi_for_plot, theta * 180 / np.pi, np.abs(to_plot[1]), **kwargs)
    plt.title(title)
    plot_spherical_grid()
    set_lims()

    plt.subplot(n_rows, 3, 3)
    plt.tricontourf(phi_for_plot, theta * 180 / np.pi, np.abs(to_plot[2]), **kwargs)
    plot_spherical_grid()
    set_lims()

    if comp_to_plot is not None:
        plt.subplot(n_rows, 3, 4)
        args = (phi_for_plot, theta * 180 / np.pi, np.abs(comp_to_plot[0]))
        plt.tricontourf(*args, **kwargs)
        if save_plot_data_filename is not None:
            np.savetxt(
                f"{save_plot_data_filename}_x_comp.txt",
                np.stack(args)
            )
        plt.ylabel("θ (°)")
        plt.xlabel("φ (°)")
        plot_spherical_grid()
        set_lims()

        plt.subplot(n_rows, 3, 5)
        plt.tricontourf(phi_for_plot, theta * 180 / np.pi, np.abs(comp_to_plot[1]), **kwargs)
        plt.xlabel("φ (°)")
        plt.title("Measurement")
        plot_spherical_grid()
        set_lims()

        plt.subplot(n_rows, 3, 6)
        plt.tricontourf(phi_for_plot, theta * 180 / np.pi, np.abs(comp_to_plot[2]), **kwargs)
        plt.xlabel("φ (°)")
        plt.colorbar()
        plot_spherical_grid()
        set_lims()


def find_optimal_parameters(meep=None, case=None, max_order=2, cartesian=True, plot_during=False, comsol_direct=False,
                            compensate_symmetry=False, print_solution=False):
    plt.close("all")
    print(f"Running {meep=}, {case=}, {max_order=}, {cartesian=}", end=" ")

    data_ref_cartesian = {}
    data_ref_spherical = {}
    theta, phi, e_r, e_theta, e_phi = (None, ) * 5
    for _, mx, my, mz, _ in iterate_over_cartesian(max_order):
        _, _, _, r, theta, phi, e_r, e_theta, e_phi = import_comsol(mx, my, mz, case=f"mathematica/{case}")
        data_ref_cartesian[(mx, my, mz)] = np.array((e_r, e_theta, e_phi))
    for _, mx, my, mz, coeff, l, m in iterate_over_spherical_then_cartesian(max_order, return_lm=True):
        if (l, m) not in data_ref_spherical:
            data_ref_spherical[(l, m)] = 0.
        data_ref_spherical[(l, m)] = data_ref_cartesian[(mx, my, mz)] * coeff + data_ref_spherical[(l, m)]
    for _, mx, my, mz, _ in iterate_over_cartesian(max_order):
        data_ref_cartesian[(mx, my, mz)] = data_ref_cartesian[(mx, my, mz)] / \
                                           np.max(np.abs(data_ref_cartesian[(mx, my, mz)]))
    for _, l, m, _ in iterate_over_spherical(max_order):
        data_ref_spherical[(l, m)] = data_ref_spherical[(l, m)] / \
                                           np.max(np.abs(data_ref_spherical[(l, m)]))

    if not comsol_direct:
        with open(f"data/meep/{meep}.pickle", "rb") as fd:
            theta_meas, phi_meas, e_r_meas, e_theta_meas, e_phi_meas = pickle.load(fd)
    else:
        if "_000_" in meep:
            _, _, _, _, theta_meas, phi_meas, e_r_meas, e_theta_meas, e_phi_meas = import_comsol(
                0, 0, 0, case=f"mathematica/{case}", compensate_symmetry=compensate_symmetry
            )
        else:
            _, _, _, _, theta_meas, phi_meas, e_r_meas, e_theta_meas, e_phi_meas = import_comsol(
                0, 0, 2, case=f"mathematica/{case}", compensate_symmetry=compensate_symmetry)
        if "comsol" in case:
            e_r_meas, e_theta_meas, e_phi_meas = (np.conj(ei) for ei in (e_r_meas, e_theta_meas, e_phi_meas))
    meas_max = max(np.max(np.abs(e_r_meas)), np.max(np.abs(e_theta_meas)), np.max(np.abs(e_phi_meas)))
    e_r_meas, e_theta_meas, e_phi_meas = e_r_meas / meas_max, e_theta_meas / meas_max, e_phi_meas / meas_max
    kwargs_interp = dict(method="nearest", fill_value=0)
    e_r_meas = griddata(
        np.stack((theta_meas, phi_meas)).T, e_r_meas,
        np.stack((theta, phi)).T,
        **kwargs_interp
    )
    e_theta_meas = griddata(
        np.stack((theta_meas, phi_meas)).T, e_theta_meas,
        np.stack((theta, phi)).T,
        **kwargs_interp
    )
    e_phi_meas = griddata(
        np.stack((theta_meas, phi_meas)).T, e_phi_meas,
        np.stack((theta, phi)).T,
        **kwargs_interp
    )

    print("... finished importing and interpolating", end=" ")
    if cartesian:
        n_moments = len(data_ref_cartesian)
    else:
        n_moments = len(data_ref_spherical)

    total_norm = norm(theta, e_r_meas, e_theta_meas, e_phi_meas)
    figsize_result = (20, 7)
    if plot_during:
        plt.figure(figsize=figsize_result)

    n_called = 0

    def get_error(x, plot=plot_during, plt_wait=True, save_plot_filename=None):
        nonlocal n_called
        total_field = (0, 0, 0)
        if cartesian:
            for _ind, _mx, _my, _mz, _ in iterate_over_cartesian(max_order):
                _e_r, _e_theta, _e_phi = data_ref_cartesian[(_mx, _my, _mz)]
                amplitude = (x[_ind] + 1j * x[len(x) // 2 + _ind])
                total_field = amplitude * _e_r + total_field[0], \
                              amplitude * _e_theta + total_field[1], \
                              amplitude * _e_phi + total_field[2]
        else:
            for _ind, _l, _m, _ in iterate_over_spherical(max_order):
                _e_r, _e_theta, _e_phi = data_ref_spherical[(_l, _m)]
                amplitude = (x[_ind] + 1j * x[len(x) // 2 + _ind])
                total_field = amplitude * _e_r + total_field[0], \
                              amplitude * _e_theta + total_field[1], \
                              amplitude * _e_phi + total_field[2]

        error = norm(theta,
                     (total_field[0]) - e_r_meas,
                     (total_field[1]) - e_theta_meas,
                     (total_field[2]) - e_phi_meas) / total_norm

        if plot and n_called % 100 == 0:
            plot_spherical_field(theta, phi, total_field, e_comp=(e_r_meas, e_theta_meas, e_phi_meas),
                                 title=f"{meep}, residual error = {error ** .5 * 100:.2f} %, prediction",
                                 save_plot_data_filename=save_plot_filename)

            if plt_wait:
                plt.pause(.0001)

        n_called += 1

        return error

    # x0 = np.zeros((n_moments, ))
    np.random.seed(0)
    x0 = (np.random.random((2 * n_moments, )) * 2 - 1.) / n_moments
    res = minimize(
        get_error,
        x0,
        tol=1e-8,
        options=dict(disp=False)
    )
    res_name = f"{meep}_{case}_{max_order}_{'cartesian' if cartesian else 'spherical'}".replace('/', '_')
    amplitudes = res.x[:n_moments] + 1j * res.x[n_moments:]
    max_size = np.max(np.abs(amplitudes))

    # print(f"{np.abs(amplitudes)=}")

    def get_size(ind_):
        return np.sqrt((np.abs(amplitudes[ind_]) / max_size) / np.pi) * 2 * 20 + 2

    if print_solution:
        print("Moments:")

    fig = plt.figure(figsize=(2.5, 2.5))
    data = []
    if cartesian:
        axis = fig.add_subplot(projection='3d')
        axis.view_init(elev=30, azim=45, roll=0)
        axis.set_proj_type("ortho")
        axis.xaxis.pane.fill = False
        axis.yaxis.pane.fill = False
        axis.zaxis.pane.fill = False
        axis.xaxis.pane.set_edgecolor('w')
        axis.yaxis.pane.set_edgecolor('w')
        axis.zaxis.pane.set_edgecolor('w')
        for ind, ax, ay, az, _ in iterate_over_cartesian(max_order):
            order = ax + ay + az
            size = get_size(ind)
            color = plt.get_cmap("autumn")(1 - order/(max_order + 1))
            axis.plot(ax, ay, ".", zs=az, markersize=size,
                      color=color)
            data.append((ax, ay, az, size))
            axis.plot((ax, ax, ), (ay, ay, ), (-.1, az, ), "-", color=color, zorder=max_order - order,
                      alpha=.2)
            if print_solution:
                print(ax, ay, az, f"\t{np.abs(amplitudes[ind]):7.05f}", "<-->", amplitudes[ind])
        axis.set_xlim(0, max_order)
        axis.set_ylim(0, max_order)
        axis.set_zlim(0, max_order)
        axis.set_xlabel("ax")
        axis.set_ylabel("ay")
        axis.set_zlabel("az")
    else:
        axis = fig.add_subplot()
        ind = 0
        for l in range(0, max_order + 1):
            for m in range(-l, l + 1):
                order = l
                size = get_size(ind)
                color = plt.get_cmap("autumn")(1 - order/(max_order + 1))
                axis.plot(m, l, ".", markersize=size,
                          color=color)
                data.append((m, l, size))
                if print_solution:
                    print(l, m, f"\t{np.abs(amplitudes[ind]):7.05f}", "<-->", amplitudes[ind])
                ind += 1
        axis.set_ylim(-.2, max_order + .2)
        axis.set_xlim(-max_order - .2, max_order + .2)
        axis.set_xlabel("m")
        axis.set_ylabel("l")
        plt.grid()

    np.savetxt(
        f"figs/{res_name}_moments_data.txt",
        np.array(data),
        delimiter=","
    )

    plt.tight_layout()
    plt.savefig(f"figs/{res_name}_moments.pdf")

    plt.figure(figsize=figsize_result)
    n_called = 0
    residual_error = get_error(res.x, plot=True, plt_wait=False, save_plot_filename=f"figs/{res_name}_fields_data")
    plt.tight_layout()
    plt.savefig(f"figs/{res_name}_fields.pdf")

    print(f"... done, residual={res.fun**.5*100:.3f} % = {10*np.log10(res.fun**.5):.3f} dB")
    if not plot_during:
        plt.close("all")
    # if cartesian:
    #     plt.figure()
    #     plot_moments(amplitudes, max_order=max_order)
    return residual_error**.5


def main():
    eps = 4
    res = 50
    compensate_symmetry = True
    find_optimal_parameters(meep=f"sim_results_000_res_{res}_eps_{eps}", case=f"eps_{eps}",
                            plot_during=True,
                            comsol_direct=False, max_order=2, cartesian=False,
                            compensate_symmetry=compensate_symmetry, print_solution=True)
    find_optimal_parameters(meep=f"sim_results_000_res_{res}_eps_{eps}", case=f"eps_{eps}",
                            plot_during=True,
                            comsol_direct=False, max_order=2, cartesian=True,
                            compensate_symmetry=compensate_symmetry, print_solution=True)
    find_optimal_parameters(meep=f"sim_results_200_res_{res}_eps_{eps}", case=f"eps_{eps}",
                            plot_during=True,
                            comsol_direct=False, max_order=2, cartesian=False,
                            compensate_symmetry=compensate_symmetry, print_solution=True)
    find_optimal_parameters(meep=f"sim_results_200_res_{res}_eps_{eps}", case=f"eps_{eps}",
                            plot_during=True,
                            comsol_direct=False, max_order=2, cartesian=True,
                            compensate_symmetry=compensate_symmetry, print_solution=True)
    plot_all_cartesian_results(1, f"mathematica/eps_{eps}")
    plot_all_spherical_results(1, f"mathematica/eps_{eps}")
    plt.show()
    # do_sweep()
    # plot_sweep_results()
    # do_all_moments_plots()


def do_all_moments_plots():
    eps_zz = 4
    case = f"eps_{eps_zz}"
    res = 50
    max_order = 3
    plot = True
    find_optimal_parameters(
        meep=f"sim_results_000_res_{res}_eps_{eps_zz}",
        case=case, plot_during=plot,
        comsol_direct=False, max_order=max_order, cartesian=False,
        compensate_symmetry=False
    ),
    find_optimal_parameters(
        meep=f"sim_results_000_res_{res}_eps_{eps_zz}", case=case,
        plot_during=plot,
        comsol_direct=False, max_order=max_order,
        compensate_symmetry=False
    )
    find_optimal_parameters(
        meep=f"sim_results_200_res_{res}_eps_{eps_zz}",
        case=case, plot_during=plot,
        comsol_direct=False, max_order=max_order, cartesian=False,
        compensate_symmetry=False
    ),
    find_optimal_parameters(
        meep=f"sim_results_200_res_{res}_eps_{eps_zz}", case=case,
        plot_during=plot,
        comsol_direct=False, max_order=max_order,
        compensate_symmetry=False
    )
    plt.show()


def do_sweep():
    for max_order in range(7):
        for eps_zz in (1, 2, 3, 4, 5, 6, 8, 10, 18, 32, 56, 100, 200, 1000, 10000, ):
            print(f"--- {eps_zz=}, {max_order=}")
            res = inverse_problem(eps_zz, max_order=max_order, run_meep=False, inverse_crime=eps_zz > 10)
            with open(f"results/sweep/eps_{eps_zz}_max-order_{max_order}.pickle", "wb") as fd:
                pickle.dump(res, fd)


def plot_sweep_results():
    results = dict()
    eps_zzs = (1, 2, 3, 4, 5, 6, 8, 10, 18, 32, 56, 100, 200, 1000, )  # 10_000, )
    max_orders = list(range(6 + 1))
    for eps_zz in eps_zzs:
        for max_order in max_orders:
            filename = f"results/sweep/eps_{eps_zz}_max-order_{max_order}.pickle"
            with open(filename, "rb") as fd:
                results[(eps_zz, max_order)] = pickle.load(fd)

    eps_zz = 4
    fig, ax1 = plt.subplots(figsize=(6, 4))
    color1, color2 = "darkblue", "royalblue"
    color3, color4 = "darkslategray", "darkturquoise"
    y_000_spherical = [100 - results[(eps_zz, m)][("000", "spherical")] * 100 for m in max_orders]
    y_000_cartesian = [100 - results[(eps_zz, m)][("000", "cartesian")] * 100 for m in max_orders]
    y_200_spherical = [100 - results[(eps_zz, m)][("200", "spherical")] * 100 for m in max_orders]
    y_200_cartesian = [100 - results[(eps_zz, m)][("200", "cartesian")] * 100 for m in max_orders]
    plt.plot(
        max_orders,
        y_000_spherical, "-", color=color1,
        label="000, spherical"
    )
    plt.plot(
        max_orders,
        y_000_cartesian, "--", color=color2,
        label="000, Cartesian"
    )
    plt.plot(
        max_orders,
        y_200_spherical, "-o", color=color1,
        label="200, spherical"
    )
    plt.plot(
        max_orders,
        y_200_cartesian, "-d", color=color2,
        label="200, cartesian"
    )
    plt.xlabel("Truncation order")
    plt.title(f"eps_zz = {eps_zz}")
    plt.ylabel("% explained ground truth")
    plt.legend()
    plt.xlim(0, max(max_orders))
    plt.ylim(0, 102)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    n = np.array(max_orders)
    n_spherical = [
            sum([2 * i + 1 for i in range(0, ni + 1)]) for ni in n
        ]
    n_cartesian = [
            sum([(i + 1) * (i + 2) / 2 for i in range(0, ni + 1)]) for ni in n
        ]
    ax2.plot(
        n, n_spherical, "-.", label="Spherical", color=color3
    )
    ax2.plot(
        n, n_cartesian, ":",
        label="Cartesian", color=color4
    )
    plt.legend()
    ax2.tick_params(axis='y')
    ax2.set_ylim(0, 120)
    ax2.set_ylabel("Number of degrees of freedom")

    plt.tight_layout()
    plt.savefig("figs/metrics_vs_n.pdf")
    np.savetxt("figs/data_metrics_vs_n_1.txt",
               np.stack((
                   max_orders,
                   y_000_spherical,
                   y_000_cartesian,
                   y_200_spherical,
                   y_200_cartesian
               ))
               )
    np.savetxt("figs/data_metrics_vs_n_2.txt",
               np.stack((
                    n, n_spherical, n_cartesian
               ))
               )

    plt.figure(figsize=(4, 3))
    # lines = ["-", "--", ":", "-.", "-:"]
    for ind, max_order in enumerate(max_orders):
        if ind % 2 == 1 or max_order < 2:
            continue
        d = [100 - results[(eps_zz, max_order)][("200", "spherical")] * 100 for eps_zz in eps_zzs]
        plt.semilogx(
            eps_zzs,
            d, f"-",
            color=plt.get_cmap("autumn")(1 - ind/len(max_orders)),
            label=f"l_max={max_order}"
        )
        plt.plot(
            eps_zzs,
            d, f".",
            color=plt.get_cmap("autumn")(1 - ind/len(max_orders)),
            label=f"l_max={max_order}"
        )
        np.savetxt(
            f"figs/metrics_vs_eps_zz_max_order_{max_order}.txt",
            np.stack((eps_zzs, d))
        )

    plt.xlabel("eps_zz")
    plt.ylabel("% explained ground truth")
    plt.legend()
    plt.ylim(60, 100)
    plt.xlim(min(eps_zzs), max(eps_zzs))
    plt.grid()

    plt.tight_layout()
    plt.savefig("figs/metrics_vs_eps_zz.pdf")

    # plt.figure(figsize=(6, 4))
    # plt.plot(
    #     n,
    #     [
    #         sum([2 * i + 1 for i in range(0, ni + 1)]) for ni in n
    #     ], "b-",
    #     label="spherical"
    # )
    # plt.plot(
    #     n,
    #     [
    #         sum([(i + 1) * (i + 2) / 2 for i in range(0, ni + 1)]) for ni in n
    #     ], "k--",
    #     label="Cartesian"
    # )
    # plt.xlabel("Truncation order")
    # plt.title(f"Number of degrees of freedom")
    # plt.legend()
    # plt.xlim(0, max(max_orders))
    # # plt.ylim(40, 100)
    # plt.grid()
    # plt.tight_layout()

    plt.show()


def inverse_problem(eps_zz, max_order=0, plot=False, run_meep=False, inverse_crime=False):
    # plot_all_results(2, "mathematica/eps_10")
    # return
    if plot:
        plt.figure()
        plt.close(1)
    res = {
        1: {False: 50, True: 50},
        1.5: {False: 50, True: 50},
        2: {False: 50, True: 50},
        3: {False: 50, True: 50},
        4: {False: 50, True: 50},
        5: {False: 70, True: 70},
        6: {False: 100, True: 200},
        8: {False: 120, True: 120},
        10: {False: 170, True: 170},
        18: {False: -1, True: -1},
        32: {False: -1, True: -1},
        56: {False: -1, True: -1},
        100: {False: -1, True: -1},
        200: {False: -1, True: -1},
        1_000: {False: -1, True: -1},
        10_000: {False: -1, True: -1}
    }[eps_zz]
    if run_meep:
        run_simulation(
            case=f"mathematica/eps_{eps_zz}",
            diag=(1, 1, eps_zz),
            d2x=False,
            resolution=res[False],
            save_name=f"sim_results_000_res_{res[False]}_eps_{eps_zz}"
        )
        run_simulation(
            case=f"mathematica/eps_{eps_zz}",
            diag=(1, 1, eps_zz),
            d2x=True,
            resolution=res[True],
            save_name=f"sim_results_200_res_{res[True]}_eps_{eps_zz}"
        )
    comsol_direct = inverse_crime
    compensate_symmetry = False
    case = f"eps_{eps_zz}"
    results = {
        ("000", "spherical"): find_optimal_parameters(meep=f"sim_results_000_res_{res[False]}_eps_{eps_zz}", case=case,
                                                      plot_during=plot,
                                                      comsol_direct=comsol_direct, max_order=max_order, cartesian=False,
                                                      compensate_symmetry=compensate_symmetry),
        ("200", "spherical"): find_optimal_parameters(meep=f"sim_results_200_res_{res[True]}_eps_{eps_zz}",
                                                      case=case, plot_during=plot,
                                                      comsol_direct=comsol_direct, max_order=max_order, cartesian=False,
                                                      compensate_symmetry=compensate_symmetry),
        ("000", "cartesian"): find_optimal_parameters(meep=f"sim_results_000_res_{res[False]}_eps_{eps_zz}", case=case,
                                                      plot_during=plot,
                                                      comsol_direct=comsol_direct, max_order=max_order,
                                                      compensate_symmetry=compensate_symmetry),
        ("200", "cartesian"): find_optimal_parameters(meep=f"sim_results_200_res_{res[True]}_eps_{eps_zz}", case=case,
                                                      plot_during=plot,
                                                      comsol_direct=comsol_direct, max_order=max_order,
                                                      compensate_symmetry=compensate_symmetry)
    }
    if plot:
        # plt.close("all")
        plt.show()
    return results


def meep_amp_func(x, sigma=1., d2x=False):
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    if not d2x:
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1/2*(r / sigma)**2)
    else:
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1/2*(r / sigma)**2) * (x[2]**2 - sigma**2) / sigma**4


def plot_moments(moments, max_order=2):
    plot_ind = 1
    moment_ind = 0
    max_moments = np.max(np.abs(moments))
    for _, mx, my, mz, _ in iterate_over_cartesian(max_order):
        amp = moments[moment_ind]
        plt.subplot(max_order + 1, 1, mx + my + mz + 1)
        plt.plot(
            mx, my, '.',
            markersize=20 * np.abs(amp) / max_moments,
            color=plt.get_cmap("jet")(np.abs(amp) / max_moments)
        )
        plt.xlim(-1, max_order + 1)
        plt.ylim(-1, max_order + 1)
        moment_ind += 1

        plot_ind += 1


def meep_simulation(x, y, z, resolution=10, d2x=False, diag=(1, 1, 10), save_name=None):
    if save_name is None:
        save_name = "sim"
    c_si = 1
    f_si = 1
    f = f_si / c_si

    r_omega = 1.2
    dr = 0.05
    r_meas = 1

    print(f"Transverse number of points in the source region: {int(2 * dr * resolution)}")

    cell_size = mp.Vector3(2 * r_omega, 2 * r_omega, 2 * r_omega)

    geometry = [mp.Block(mp.Vector3(2 * r_omega, 2 * r_omega, 2 * r_omega),
                         material=mp.Medium(epsilon_diag=diag)),
                ]

    source_width = 0.01
    if not d2x:
        sources = [
            mp.Source(
                mp.ContinuousSource(f, width=source_width),
                mp.Ex,
                amplitude=1,
                center=(0, 0, 0),
            )
        ]
    else:
        step = 2 * r_omega / resolution
        # step = 2 * r_omega / 30
        end_point = step * (4 - 1) / 2
        sources = []
        for amp, pos_source in zip((1, -1, -1, 1), np.linspace(-end_point, end_point, 4)):
            sources.append(
                mp.Source(
                    mp.ContinuousSource(f, width=source_width),
                    mp.Ex,
                    amplitude=amp,
                    center=(0, 0, pos_source),
                )
            )

    symmetries = [
        mp.Mirror(mp.X, -1),
        mp.Mirror(mp.Y, 1),
        mp.Mirror(mp.Z, 1)
    ]
    pml_layers = [mp.PML(.5 * (r_omega - r_meas))]
    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        sources=sources,
        symmetries=symmetries,
        boundary_layers=pml_layers,
        resolution=resolution,
        force_complex_fields=True
    )
    sim.init_sim()
    sim.solve_cw(1e-6, 10_000, 2)

    ex, ey, ez = get_meep_field_on_sphere(sim, x, y, z)
    theta, phi, e_r, e_theta, e_phi = get_fields_on_sphere(x, y, z, ex, ey, ez)
    plot_all(0, 0, 0, theta, phi, e_r, e_theta, e_phi)

    with open(f"data/meep/{save_name}.pickle", "wb") as fd:
        pickle.dump((theta, phi, e_r, e_theta, e_phi), fd)

    return theta, phi, e_r, e_theta, e_phi


def get_meep_field_on_sphere(sim, x, y, z):
    ex, ey, ez = np.zeros(x.size, dtype="complex"), \
                    np.zeros(x.size, dtype="complex"), \
                    np.zeros(x.size, dtype="complex")
    for ind, (xi, yi, zi) in enumerate(zip(x, y, z)):
        args = ((xi, yi, zi), )
        ex[ind] = sim.get_field_point(mp.Ex, *args)
        ey[ind] = sim.get_field_point(mp.Ey, *args)
        ez[ind] = sim.get_field_point(mp.Ez, *args)

    return ex, ey, ez


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")
    main()
