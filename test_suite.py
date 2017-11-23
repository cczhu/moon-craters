import unittest
import make_input_data as mkin
import re
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.img_transform as cimg
from PIL import Image
from scipy import signal


class CatalogueTest(unittest.TestCase):
    """Tests crater catalogues."""

    def setUp(self):

        # Head et al. dataset.
        self.lrochead = mkin.ReadLROCHeadCombinedCraterCSV(sortlat=True)

        # Craters with identical latitudes are s
        head = pd.read_csv('./HeadCraters.csv', header=0,
                           names=['Long', 'Lat', 'Diameter (km)'])
        lroc = pd.read_csv('./LROCCraters.csv', usecols=range(2, 5), header=0)

        self.lrochead_t = pd.concat([lroc, head], axis=0, ignore_index=True,
                                    copy=True)
        self.lrochead_t.sort_values(by='Lat', inplace=True)
        self.lrochead_t.reset_index(inplace=True, drop=True)

        # Salamuniccar et al. dataset.
        self.lroclu = mkin.ReadLROCLUCombinedCraterCSV(dropfeatures=True)

        sala_names = ("Long", "Lat", "Diameter (km)", "Name")
        sala_types = (float, float, float, str)
        sala = pd.read_csv('./LU78287GT.csv', usecols=(1, 2, 4, 7), header=0,
                           engine="c", encoding="ISO-8859-1", names=sala_names,
                           dtype=dict(zip(sala_names, sala_types)))
        sala["Name"] = sala["Name"].str.split(":").str.get(0)

        def match_end(s):
            if re.match(r" [A-Z]", s[-2:]):
                return True
            return False

        tocut = sala.loc[sala["Name"].notnull(), "Name"].apply(match_end)
        tocut = tocut[tocut].index
        sala.drop(tocut, inplace=True)
        sala.drop(["Name"], axis=1, inplace=True)
        sala = sala[sala["Diameter (km)"] > 20]

        self.lroclu_t = pd.concat([sala, lroc], axis=0, ignore_index=True,
                                  copy=True)

        self.lroclu_t.sort_values(by='Lat', inplace=True)
        self.lroclu_t.reset_index(inplace=True, drop=True)

    def tearDown(self):
        self.lrochead = None
        self.lrochead_t = None
        self.lroclu = None
        self.lroclu_t = None

    def test_dataframes_equal(self):
        self.assertTrue(self.lrochead.equals(self.lrochead_t))
        self.assertTrue(self.lroclu.equals(self.lroclu_t))


class CoordTest(unittest.TestCase):
    """Tests pix2coord and coord2pix
    """

    def setUp(self):

        origin = np.random.uniform(-30, 30, 2)
        extent = np.random.uniform(0, 45, 2)
        self.cdim = [origin[0], origin[0] + extent[0],
                     origin[1], origin[1] + extent[1]]
        self.imgdim = np.random.randint(100, high=200, size=2)

        self.cx = np.array(
            [self.cdim[1], np.random.uniform(self.cdim[0] + 1, self.cdim[1])])
        self.cy = np.array(
            [self.cdim[3], np.random.uniform(self.cdim[2] + 1, self.cdim[3])])

    def test_coord2pix(self):

        x_gt = (self.imgdim[0] *
                (self.cx - self.cdim[0]) / (self.cdim[1] - self.cdim[0]))
        y_gt = (self.imgdim[1] *
                (self.cy - self.cdim[2]) / (self.cdim[3] - self.cdim[2]))
        yi_gt = (self.imgdim[1] *
                 (self.cdim[3] - self.cy) / (self.cdim[3] - self.cdim[2]))

        for origin in ["lower", "upper"]:
            with self.subTest(origin=origin):
                x, y = mkin.coord2pix(self.cx, self.cy, self.cdim,
                                      self.imgdim, origin=origin)
                if origin == "upper":
                    y_gt_curr = yi_gt
                else:
                    y_gt_curr = y_gt
                xy = np.r_[x, y]
                xy_gt = np.r_[x_gt, y_gt_curr]
                self.assertTrue(np.all(
                    np.isclose(xy, xy_gt, rtol=1e-7, atol=1e-10)))

    def test_pix2coord(self):

        for origin in ["lower", "upper"]:
            with self.subTest(origin=origin):
                x, y = mkin.coord2pix(self.cx, self.cy, self.cdim,
                                      self.imgdim, origin=origin)
                cx, cy = mkin.pix2coord(x, y, self.cdim, self.imgdim,
                                        origin=origin)
                cxy = np.r_[cx, cy]
                cxy_gt = np.r_[self.cx, self.cy]
                self.assertTrue(np.all(
                    np.isclose(cxy, cxy_gt, rtol=1e-7, atol=1e-10)))

    def test_km2pix(self):

        mykmppix = 1500. / (np.pi * 1737.4) * 0.5
        kmppix = mkin.km2pix(1500., 180., dc=0.5, a=1737.4)

        self.assertTrue(np.isclose(mykmppix, kmppix, rtol=1e-7, atol=1e-10))


class WarpTest(unittest.TestCase):

    def setUp(self):

        self.img = Image.open(
            'LunarLROLrocKaguya_1180mperpix_downsamp.png').convert("L")
        imgsize = list(self.img.size)
        self.img = self.img.crop([0, 0, 300, 300])
        self.img = self.img.resize([200, 200])

        # Take top edge of long-lat bounds
        ix = np.array([0, 300])
        iy = np.array([0, 300])
        cdim = [-180, 180, -60, 60]
        llong, llat = mkin.pix2coord(ix, iy, cdim, 
                            imgsize, origin="upper")
        self.llbd = np.r_[llong, llat[::-1]]

        self.iglobe = ccrs.Globe(semimajor_axis=1737400, 
                        semiminor_axis=1737400,
                        ellipse=None)

        self.geoproj = ccrs.Geodetic(globe=self.iglobe)
        self.iproj = ccrs.PlateCarree(globe=self.iglobe)
        self.oproj = ccrs.Orthographic(central_longitude=np.mean(self.llbd[:2]), 
                                central_latitude=np.mean(self.llbd[2:]), 
                                globe=self.iglobe)

        xllr = np.array([self.llbd[0], np.mean(self.llbd[:2]), 
                            self.llbd[1]])
        yllr = np.array([self.llbd[2], np.mean(self.llbd[2:]), 
                            self.llbd[3]])
        xll, yll = np.meshgrid(xllr, yllr)
        xll = xll.ravel(); yll = yll.ravel()

        # [:,:2] becaus we don't need elevation data
        res = self.iproj.transform_points(x=xll, y=yll,
                                src_crs=self.geoproj)[:,:2]
        self.iextent = [min(res[:,0]), max(res[:,0]), 
                    min(res[:,1]), max(res[:,1])]

        res = self.oproj.transform_points(x=xll, y=yll,
                                src_crs=self.geoproj)[:,:2]
        self.oextent = [min(res[:,0]), max(res[:,0]), 
                    min(res[:,1]), max(res[:,1])]

        self.craters = pd.DataFrame(np.vstack([xllr, yllr]).T, 
                                columns=["Long", "Lat"])
        self.craters["Diameter (km)"] = [10, 10, 10]


    def tearDown(self):
        self.img = None


    def test_warpimage(self):

        img = np.asanyarray(self.img)
        img = img[::-1]

        regrid_shape = 1.2*min(img.shape)
        regrid_shape = mkin.regrid_shape_aspect(regrid_shape,
                                         self.oextent)

        imgout, ext = cimg.warp_array(np.asanyarray(self.img),
                             source_proj=self.iproj,
                             source_extent=self.iextent,
                             target_proj=self.oproj,
                             target_res=regrid_shape,
                             target_extent=self.oextent,
                             mask_extrapolated=True)

        imgout = np.ma.filled(imgout[::-1], fill_value=0)

        imgoutmkin = mkin.WarpImage(img, self.iproj, 
                        self.iextent, self.oproj, self.oextent,
                        origin="upper", rgcoeff=1.2)
        imgoutmkin = np.ma.filled(imgoutmkin, fill_value=0)

        self.assertTrue( np.all(np.isclose(imgout.ravel(), 
                        imgoutmkin.ravel(), 
                        rtol=1e-6, atol=1e-10)) )


    def test_warpcraters(self):

        # Not the real image dimensions, but whatever
        imgdim = [250, 300]

        ilong = self.craters["Long"].as_matrix()
        ilat = self.craters["Lat"].as_matrix()
        res = self.oproj.transform_points(x=ilong, y=ilat,
                                src_crs=self.geoproj)[:,:2]

        # Get output
        x, y = mkin.coord2pix(res[:,0], 
                        res[:,1], self.oextent, imgdim, 
                        origin="upper")

        ctr_sub = mkin.WarpCraterLoc(self.craters, self.geoproj, self.oproj, 
                        self.oextent, imgdim, llbd=None,
                        origin="upper")

        xy_gt = np.r_[x, y]
        xy = np.r_[ctr_sub["x"].as_matrix(), ctr_sub["y"].as_matrix()]

        self.assertTrue( np.all(np.isclose(xy, xy_gt, 
                        rtol=1e-7, atol=1e-10)) )


    def test_pctoortho(self):

        imgo, imgwshp, offset = mkin.WarpImagePad(self.img, self.iproj, self.iextent, 
                        self.oproj, self.oextent, origin="upper", rgcoeff=1.2, 
                        fillbg="black")

        ctr_xy = mkin.WarpCraterLoc(self.craters, self.geoproj, self.oproj, 
                        self.oextent, imgwshp, llbd=None,
                        origin="upper")

        ctr_xy.loc[:, "x"] += offset[0]
        ctr_xy.loc[:, "y"] += offset[1]

        Cd = 1.
        pxperkm = mkin.km2pix(imgo.size[1], self.llbd[3] - self.llbd[2], \
                            dc=Cd, a=1737.4)
        ctr_xy["Diameter (pix)"] = ctr_xy["Diameter (km)"] * pxperkm

        imgo2, ctr_xy2 = mkin.PlateCarree_to_Orthographic(self.img, None, self.llbd, 
                                self.craters, iglobe=self.iglobe, ctr_sub=True, 
                                origin="upper", rgcoeff=1.2,
                                dontsave=True, slivercut=0.)

        imgo = np.asanyarray(imgo)
        imgo2 = np.asanyarray(imgo2)

        self.assertTrue( np.all(np.isclose(imgo.ravel(), 
                        imgo2.ravel(), 
                        rtol=1e-6, atol=1e-10)) )
        self.assertTrue( ctr_xy.equals(ctr_xy2) )


class DensMapTest(unittest.TestCase):

    def setUp(self):

        self.img =  np.asanyarray(Image.open("moonmap_tiny.png").convert("L"))
        self.craters = mkin.ReadCombinedCraterCSV(dropfeatures=True)
        cx, cy = mkin.coord2pix(self.craters["Long"].as_matrix(), 
                    self.craters["Lat"].as_matrix(), 
                    [-180, 180, -90, 90],
                    [self.img.shape[1], self.img.shape[0]])
        self.craters["x"] = cx
        self.craters["y"] = cy
        self.craters["Diameter (pix)"] = \
            self.craters["Diameter (km)"]*mkin.km2pix(self.img.shape[0], 180)
        self.craters.drop( np.where(self.craters["Diameter (pix)"] < 15.)[0], inplace=True )
        self.craters.reset_index(inplace=True)

        self.img2 = np.zeros([200,200])
        crat_x_list = np.array([100, 50, 2, 4, 167, 72, 198, 1])
        crat_y_list = np.array([100, 50, 1, 191, 3, 199, 198, 199])
        self.craters2 = pd.DataFrame([crat_x_list, crat_y_list]).T.rename(columns={0 : "x", 1 : "y"})

        for i in range(len(crat_x_list)):
            self.img2[crat_y_list[i],crat_x_list[i]] +=1 # add one at crater location


    @staticmethod
    def gkern(l=5, sig=1.):
        """
        Creates Gaussian kernel with side length l and a sigma of sig
        """
        ax = np.arange(-l // 2 + 1., l // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))
        return kernel / np.sum(kernel)

    def test_make_dens_map(self):

        # Tiny Moon image
        kernel_sig = 4
        kernel_extent = 8

        dmap_delta = np.zeros(self.img.shape)

        for i in range(self.craters.shape[0]):
            dmap_delta[int(self.craters.loc[i,"y"]), int(self.craters.loc[i,"x"])] +=1 # add one at crater location

        # keep kernel support odd number (for comparison with my function)
        kernel_support = int(kernel_extent*kernel_sig/2)*2 + 1
        kernel = self.gkern(kernel_support, kernel_sig)

        img_dm_c2 = signal.convolve2d(dmap_delta, kernel, boundary='fill', mode='same')
        img_dm = mkin.make_density_map(self.craters, self.img.shape, k_sig=kernel_sig, k_support=kernel_extent)

        self.assertTrue( np.isclose(img_dm_c2, img_dm, rtol=1e-05, atol=1e-08).sum() / img_dm.size )

        # Edge cases image
        kernel_sig = 12
        kernel_extent = 5
        kernel_support = int(kernel_extent * kernel_sig/2)*2 + 1
        kernel = self.gkern(kernel_support, kernel_sig)

        img2_dm_c2 = signal.convolve2d(self.img2, kernel, boundary='fill', mode='same')
        img2_dm = mkin.make_density_map(self.craters2, self.img2.shape, k_sig=kernel_sig, k_support=kernel_extent)

        self.assertTrue( np.isclose(img2_dm_c2, img2_dm, rtol=1e-05, atol=1e-06).sum()/ img2_dm.size )


def run_cataloguetest():
    suite = unittest.TestLoader().loadTestsFromTestCase(CatalogueTest)
    unittest.TextTestRunner(verbosity=2).run(suite)


def run_coordtest():
    suite = unittest.TestLoader().loadTestsFromTestCase(CoordTest)
    unittest.TextTestRunner(verbosity=2).run(suite)


def run_warptest():
    suite = unittest.TestLoader().loadTestsFromTestCase(WarpTest)
    unittest.TextTestRunner(verbosity=2).run(suite)


def run_dmtest():
    suite = unittest.TestLoader().loadTestsFromTestCase(DensMapTest)
    unittest.TextTestRunner(verbosity=2).run(suite)


def run_everything():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(CatalogueTest))
    suite.addTest(unittest.makeSuite(CoordTest))
    suite.addTest(unittest.makeSuite(WarpTest))
    suite.addTest(unittest.makeSuite(DensMapTest))
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    unittest.main()
