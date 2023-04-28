from mpdaf.obj import Cube
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from mpl_toolkits.mplot3d import Axes3D


def free_expansion(vel_kms, tsn_s):
    """
    Free expansion velocity model for ejected material.
    It assumes the material has expanded at the same velocity we observe today.

    :param vel_kms: numpy array
        velocities to convert to z_au, in km/s
    :param tsn_s: float
        time since nova, in seconds
    :return: numpy array with z_au values
    """

    return vel_kms * tsn_s / 1.496e8  # AU


def duerbeck87_decelerating_expansion(vel_kms, tsn_s, v_0, v_b):
    """
    Duerbeck87 decelerating shell model. Assumes v_exp(t) = v_0 - v_b*tsn.

    :param vel_kms: np.array
        numpy array with observed velocities, in km/s.
    :param tsn_s: float
        The time since the nova eruption, in seconds.
    :param v_0: float
        initial expansion velocity coefficient, in km/s.
    :param v_b: float
        decelerating parameters, in km/s^2.
    :return: np.array
        numpy array with the values for z_au.
    """
    r = v_0*tsn_s - 0.5*v_b*tsn_s*tsn_s
    v_exp = v_0 - v_b*tsn_s
    return r * vel_kms / v_exp / 1.496e8  # AU


class ThreeDVisualization:
    """
    Class to extract the nova shell from MUSE datacube, and its posterior conversion to proper 3D physical space.

    TO DO : Rewrite the save_csv function to include the comments at the beggining of the file.
            Improve the ppp conversion with adding the possibility of including custom user function.
            Improve the masking adding the posibility of consider a range of velocities also.
            Maybe add an interactive masking functionality (?).
            Making the input more friendly, being able to detect if we are passing one tuple of x,y or more than one.
    """

    def __init__(self, cube_path, star_pos_pxl, line, subcube_size=30, wave_range=40, snr_threshold=2,
                 bindata=False, binfactor=(1, 2, 2), n_neighbours=3, flux_selection=0.1):
        """

        :param cube_path: str
            string containing the path to the MUSE datacube.
        :param star_pos_pxl: list
            tuple containing the y,x position of the star within the datacube.
        :param line: float
            wavelength of the line to extract, in angstrom.
        :param subcube_size: float
            size of the subcube centered at the star position, in arcsec. Default 30.
        :param wave_range: float
            subcube wavelenghts will be delimited between rest wavelength +- wave_range/2. Default 40.
        :param snr_threshold: float
            minimum S/N ratio value that a svoxel must have to be considered part of the shell. Default 2.
        :param bindata: bool
            If True datacube will be binned before start the shell extraction.
        :param binfactor: list
            tuple containing the binned factors (z,y,x) used during binned. Default (1,2,2).
        :param n_neighbours: int
            minimum number of adjacent svoxels that a svoxel must have to be considered part of the shell. Default 3.
        :param flux_selection:
            quantile defining the minimum flux for selection. Only svoxels with higher flux will be selected.
            Default 0.1.
        """

        # load the cube,
        # from pixel position computes wcs position
        # remember star_pos_pxl and star_pos_wcs are given in y,x and dec, ra respectively

        self.cube = Cube(cube_path)
        star_pos_wcs = self.cube.wcs.pix2sky(star_pos_pxl)

        # define the dictionary with parameters
        # pxl scl is given in arcsec/pxl, the round is to avoid numerical issues
        self.params = {
            "star_pos_pxl": star_pos_pxl,
            "star_pos_wcs": star_pos_wcs[0],
            "subcube_size": subcube_size,
            "rest_line": line,
            "wave_range": wave_range,
            "snr_threshold": snr_threshold,
            "bindata": bindata,
            "binfactor": binfactor,
            "pxl_scl": np.around(self.cube.wcs.get_step() * 3600, 2),
            "wave_step": self.cube.wave.get_step(),
            "minimum_adjacent_pixels": n_neighbours,
            "minimum_flux_selection": flux_selection
        }

        # set the dataframe for later used
        self.data_df = None

        # first reduce the dimensions of the cube to the region of interest
        # centered at the star position, size reduced according to parameter size
        # and wavelength range defined by rest_line and wave_range
        self.create_subcube()

        # bin data if required
        if bindata:
            self.bin_data()

        # now we can create the background subtracted cube
        self.remove_background()

        # the snr cube can be created now
        self.snr_cube = self.cube / np.sqrt(self.cube.var)

        # select those points within the cube with snr greater that snr_threshold
        self.select_snrpoints()

        # clean the points using the minimum number of neighbours criteria
        if self.params["minimum_adjacent_pixels"] > 0:
            self.clean_points_minimum_neighbours()
        if self.params["minimum_flux_selection"] > 0:
            self.clean_points_maximum_fluxes()

        # obtain the ppv space
        self.convert_to_ppv()

    def create_subcube(self):
        """
        run mpdaf subcube to create a smaller cube centered at the star position and restricted close to the line
        of interest
        """
        center = self.params["star_pos_wcs"]
        size = self.params["subcube_size"]
        lbda_low = self.params["rest_line"] - self.params["wave_range"]/2
        lbda_up = self.params["rest_line"] + self.params["wave_range"]/2
        self.cube = self.cube.subcube(center, size, lbda=(lbda_low, lbda_up))

        # update the star_pos_pxl accordingly
        star_pos_pxl = self.cube.wcs.sky2pix(self.params["star_pos_wcs"])
        self.params["star_pos_pxl_subcube"] = star_pos_pxl[0]

    def bin_data(self):
        """
        bin the cube using the Cube rebin method
        The flux are computed as mean
        """
        self.cube = self.cube.rebin(factor=self.params["binfactor"], margin="origin")

        # the flux are meaned between 8 spaxels, so data must be multiplied by number of svoxels combined to compensate
        svoxels_binned = self.params["binfactor"][0]*self.params["binfactor"][1]*self.params["binfactor"][2]
        self.cube.data = svoxels_binned * self.cube.data

        # update pxl_pos, and pxl_scls
        star_pos_pxl = self.cube.wcs.sky2pix(self.params["star_pos_wcs"])
        self.params["star_pos_pxl_subcube"] = star_pos_pxl[0]
        self.params["pxl_scl"] = np.around(self.cube.wcs.get_step() * 3600, 2)
        self.params["wave_step"] = self.cube.wave.get_step()

    def remove_background(self):
        """
        creates a background cube to subtract from the datacube.
        The cube is created from a linear interpolation between the edges of the datacubes
        """
        cube_waverange = self.cube.wave.coord()
        blue_min = cube_waverange[0]
        blue_max = cube_waverange[2]
        red_min = cube_waverange[-3]
        red_max = cube_waverange[-1]

        blue_image = self.cube.select_lambda(blue_min, lbda_max=blue_max).mean(axis=0)
        red_image = self.cube.select_lambda(red_min, lbda_max=red_max).mean(axis=0)
        backcube = self.cube.copy()

        blue_wave = cube_waverange[1]
        red_wave = cube_waverange[-2]
        m = (red_image - blue_image) / (red_wave - blue_wave)
        m = m.data
        c = 0
        for x in cube_waverange:
            slice_cont = m * x + blue_image - m * blue_wave
            backcube[c, :, :] = slice_cont
            c += 1

        self.cube = self.cube - backcube

    def select_snrpoints(self):
        """
        select svoxels within the datacube that have snr above the given threshold
        """
        cube_data = self.cube.data
        cube_var = self.cube.var
        cube_snr = self.snr_cube.data
        sel = cube_snr > self.params["snr_threshold"]
        cube_data[~sel] = 0

        # all data with snr below threshold is zero, so I can use the nonzero function to get the rest of data
        z, y, x = cube_data.nonzero()
        f = np.zeros_like(x)
        f_e = np.zeros_like(x)
        snr = np.zeros_like(x)
        for i in range(len(x)):
            f[i] = cube_data[z[i], y[i], x[i]]
            f_e[i] = cube_var[z[i], y[i], x[i]]
            snr[i] = cube_snr[z[i], y[i], x[i]]

        # x,y and z are given with respect the star position and rest line,
        # so I need to define these values to subtract them
        xy_star = self.params["star_pos_pxl_subcube"]
        z_range = self.cube.wave.pixel(self.params["rest_line"])

        # flux are in units of erg/s/cm2/A, so I need to multiply by the width of each slice
        f = f * self.params["wave_step"]
        f_e = np.sqrt(f_e) * self.params["wave_step"]

        # set the type of units so the output will look nicer
        # f and f_e as int
        f = f.astype(int)
        f_e = f_e.astype(int)
        # x, y, z and snr as a float up to the second decimal
        x = np.around(x - xy_star[1], 2)
        y = np.around(y - xy_star[0], 2)
        z = np.around(z - z_range, 2)
        snr = np.around(snr, 2)

        # the
        to_df = {
            "x_pxl": x,
            "y_pxl": y,
            "z_pxl": z,
            "flux": f,
            "flux_e": f_e,
            "snr": snr
        }
        self.data_df = pd.DataFrame(to_df)

    def clean_points_minimum_neighbours(self):
        """
        clean points from the data, using a minimum number of neighbours criteria.
        A point will need to be close to at least n_neighbours other points to be accepted.
        """
        x = self.data_df["x_pxl"].to_numpy()
        y = self.data_df["y_pxl"].to_numpy()
        z = self.data_df["z_pxl"].to_numpy()
        # array of falses
        b = np.array(np.zeros_like(x), dtype=bool)

        for i in range(len(x)):
            sel = (np.abs(x - x[i]) == 1) & (np.abs(y - y[i]) == 1) & (np.abs(z - z[i]) == 1)
            if np.count_nonzero(sel) >= self.params["minimum_adjacent_pixels"]:
                b[i] = True

        self.data_df = self.data_df.loc[b]

    def clean_points_maximum_fluxes(self):
        """
        clean the data frame by selecting only svoxels which flux is brighter that quantile given.
        """
        flux = self.data_df["flux"].to_numpy()
        flux_lower_lim = np.quantile(flux, self.params["minimum_flux_selection"])
        sel = (flux >= flux_lower_lim)
        self.data_df = self.data_df.loc[sel]

    def convert_to_ppv(self):
        """
        reads the wcs data within the cube to obtain the PPV space
        Position-Position-Velocity space.
        """
        pxl_scl_y = self.params["pxl_scl"][0]
        pxl_scl_x = self.params["pxl_scl"][1]

        x = self.data_df["x_pxl"].to_numpy()
        y = self.data_df["y_pxl"].to_numpy()
        z = self.data_df["z_pxl"].to_numpy()

        # x/y converted to arcsec using the pxl scl
        # z converted to velocity using doppler
        # save to the data df
        self.data_df["x_arcsec"] = np.around(x * pxl_scl_x, 2)
        self.data_df["y_arcsec"] = np.around(y * pxl_scl_y, 2)
        self.data_df["z_kms"] = np.around(299792.0 * z * self.params["wave_step"] / self.params["rest_line"],
                                          0).astype(int)  # km/s

        # re arange the data frame
        self.data_df = self.data_df[["x_pxl", "y_pxl", "z_pxl",
                                     "x_arcsec", "y_arcsec", "z_kms",
                                     "flux", "flux_e", "snr"]]

    def mask_ppv(self, mask_centers, radius, unit_center="arcsec", unit_radius="arcsec"):
        """
        Create circular mask to remove points from the PPV space

        :param mask_centers: list
            A list containing (y,x) positions to be masked in the PPV (with respect the central star)
        :param radius:  float or list
            The radius of the mask. If a float is given, it is applied to all masks. If a list is provided, each entry
            correspond to the radius of the respective mask in the mask_centers list.
        :param unit_center: str or None
            Units of the masks centers. By default, it assumes to be in arcsec. If param is None, then assumes the pixel
            value.
        :param unit_radius: str or None
            Unit of the mask radius. By default, it assumes to be in arcsec. if param is None, then assumes pixels
            values.
        """
        # check if the radius param is a list or not. If it is a list, check if the len is the same as mask_centers
        if type(radius) == list:
            if len(radius) != len(mask_centers):
                print("Error: radius must be a float or a list of same length as mask_centers")
                return
            radius = np.array(radius)
        else:
            radius = radius * np.ones(len(mask_centers))

        # check the units to select which columns of the df must be used
        xdf = None
        ydf = None

        if unit_center == "arcsec":
            xdf = self.data_df["x_arcsec"].to_numpy()
            ydf = self.data_df["y_arcsec"].to_numpy()
            if unit_radius == "arcsec":
                # do nothing
                pass
            elif unit_radius is None:
                radius *= self.params["pxl_sc"][0]  # assumes the pxl scale is equal for x and y

        elif unit_center is None:
            xdf = self.data_df["x_pxl"].to_numpy()
            ydf = self.data_df["y_pxl"].to_numpy()
            if unit_radius == "arcsec":
                radius /= self.params["pxl_sc"][0]
            elif unit_radius is None:
                # do nothing
                pass

        if xdf is None or ydf is None:
            print("Error: x,y did not load properly")
            return

        sel_all = np.ones(len(xdf), dtype=bool)
        for i in range(len(mask_centers)):
            yi = mask_centers[i][0]
            xi = mask_centers[i][1]
            ri = radius[i]

            # create the selection mask
            sel_i = np.sqrt((xdf - xi)**2 + (ydf - yi)**2) > ri
            sel_all = sel_all & sel_i

        # now that all masked has been created and combined we proceed to update the dataframe
        self.data_df = self.data_df.loc[sel_all]

    def convert_to_ppp(self, distance_pc, tsn_s, v_sys_kms, expansion="free", **kwargs):
        """

        :param distance_pc: float
            distance to the system, in parsec.
        :param tsn_s: float
            time since the nova eruption, in seconds.
        :param v_sys_kms: float
            systemic velocity of the system, in km/s.
        :param expansion: str
            string specifying the model to be assumed to the velocity expansion history of the material. Currently
            implemented 3 models: free and duerbeck87.
            free: it assumes a constant expansion velocity until date using the measured velocities.
            duerbeck87: it assumes the simple decelerating model proposed by Duerbeck87, which assumes the expansion
                velocity of the material decreases linearly with time according to v(t) = v_0 - v_b*t
        :param kwargs:
            coefficients passed to the expansion model

        """
        x_arcsec = self.data_df["x_arcsec"].to_numpy()
        y_arcsec = self.data_df["y_arcsec"].to_numpy()
        z_kms = self.data_df["z_kms"].to_numpy() - v_sys_kms

        # conversion from arcsec to au
        x_au = x_arcsec * distance_pc  # x, y is linearly transformed into AU
        y_au = y_arcsec * distance_pc

        dict_func = {"free": free_expansion, "duerbeck87": duerbeck87_decelerating_expansion}

        if expansion == "free":
            z_au = dict_func["free"](z_kms, tsn_s)
        elif expansion == "duerbeck87":
            z_au = dict_func["duerbeck87"](z_kms, tsn_s, kwargs["v_0"], kwargs["v_b"])
        else:
            print("Model {} not recognized. Exit the function".format(expansion))
            return

        self.data_df["x_au"] = np.around(x_au, 0).astype(int)
        self.data_df["y_au"] = np.around(y_au, 0).astype(int)
        self.data_df["z_au"] = np.around(z_au, 0).astype(int)

        # re arange the data frame
        self.data_df = self.data_df[["x_pxl", "y_pxl", "z_pxl",
                                     "x_arcsec", "y_arcsec", "z_kms",
                                     "x_au", "y_au", "z_au",
                                     "flux", "flux_e", "snr"]]

        # update the params dict
        self.params["distance_pc"] = distance_pc
        self.params["tsn_yrs"] = round(tsn_s / 86400 / 365.25, 4)
        self.params["vsys_kms"] = v_sys_kms

    def plot_ppv(self, color="flux", ms=2):

        if color == "flux":
            c = self.data_df["flux"].to_numpy()
            c = np.log10(c)
        else:
            c = self.data_df["snr"].to_numpy()

        fig = plt.figure(figsize=(6.6, 6.6), dpi=100)
        axs = fig.add_subplot(projection="3d")

        axs.scatter(self.data_df["x_arcsec"].to_numpy(), self.data_df["y_arcsec"].to_numpy(),
                    self.data_df["z_kms"].to_numpy(), c=c, s=ms, alpha=0.75)
        axs.set_xlabel("X [arcsec]")
        axs.set_ylabel("Y [arcsec]")
        axs.set_zlabel("Z [km/s]")

        x_lim = axs.get_xlim()
        y_lim = axs.get_ylim()
        limits = np.abs(np.concatenate((x_lim, y_lim)))
        max_lim = np.amax(limits)

        axs.set_xlim([-max_lim, max_lim])
        axs.set_ylim([-max_lim, max_lim])

        plt.show()

    def plot_ppp(self, color="flux", ms=2):

        if color == "flux":
            c = self.data_df["flux"].to_numpy()
            c = np.log10(c)
        else:
            c = self.data_df["snr"].to_numpy()

        fig = plt.figure(figsize=(6.6, 6.6), dpi=100)
        axs = fig.add_subplot(projection="3d")

        axs.scatter(self.data_df["x_au"].to_numpy(), self.data_df["y_au"].to_numpy(),
                    self.data_df["z_au"].to_numpy(), c=c, s=ms, alpha=0.75)
        axs.scatter(0, 0, 0, color="red", s=10)

        x_lim = axs.get_xlim()
        y_lim = axs.get_ylim()
        z_lim = axs.get_zlim()
        limits = np.abs(np.concatenate((x_lim, y_lim, z_lim)))
        max_lim = np.amax(limits)

        axs.set_xlim([-max_lim, max_lim])
        axs.set_ylim([-max_lim, max_lim])
        axs.set_zlim([-max_lim, max_lim])

        axs.set_xlabel("X [AU]")
        axs.set_ylabel("Y [AU]")
        axs.set_zlabel("Z [AU]")

        plt.show()

    def get_total_flux(self):
        """
        return the total flux of the shell
        """
        flux = self.data_df["flux"].to_numpy()
        flux_e = self.data_df["flux_e"].to_numpy()

        total_flux = np.sum(flux)
        total_flux_e = np.sqrt(np.sum(flux_e ** 2))

        print('Shell total flux     :  {:.2e} erg/s/cm2'.format(total_flux))
        print('Shell total flux err :  {:.2e} erg/s/cm2'.format(total_flux_e))

    def save_csv(self, savepath):
        # the same as was before, but need to do it better later. per da pajita :c
        # just save the data frame into a csv file
        # add the params dict entries as a coments colum
        coments_arr = []
        for key in self.params.keys():
            coment_entry = key + ':' + str(self.params[key])
            coments_arr.append(coment_entry)
        coment_df = pd.DataFrame({'comments': coments_arr})
        to_save_df = pd.concat([self.data_df, coment_df], axis=1)
        to_save_df.to_csv(savepath, index=False)


if __name__ == "__main__":

    # Example of extraction using the MUSE datacube for RR Pic and extracting the Hb data
    rr_pic_cube = "RR_Pic/DATACUBE_RR_Pic_std.fits"

    # remember position of star must be in y/x instead of usual x/y
    rr_pic_star_pos = (166.1, 169.3)
    d = 501
    v_sys = 1.8  # Ribeiro06
    t_nova = dt.datetime(1925, 5, 25, 0, 0, 0)
    t_muse = dt.datetime(2021, 12, 12, 0, 0, 0)
    tsn = (t_muse - t_nova).total_seconds()

    print("extracting shell")
    rr_pic_hb = ThreeDVisualization(rr_pic_cube, rr_pic_star_pos, 4861.3, subcube_size=45, wave_range=40)
    rr_pic_hb.mask_ppv([[0,0]], 5)
    rr_pic_hb.convert_to_ppp(distance_pc=d, tsn_s=tsn, v_sys_kms=v_sys)
    print("plotting")
    rr_pic_hb.plot_ppp()
