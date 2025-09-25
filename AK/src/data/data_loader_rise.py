
import numpy as np

class DataLoader(object):
    def __init__(self, band_names, no_data_value=np.nan):
        self.band_names = band_names
        self.no_data_value = no_data_value

    def NDVI(self, data, v):
        band_nir = v[0]
        band_red = v[1]

        nir = data[:, :, band_nir]
        red = data[:, :, band_red]

        mask = (nir == self.no_data_value) | (red == self.no_data_value)

        ndvi = (nir - red) / (nir + red + 1e-10)
        ndvi[mask] = self.no_data_value
        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = ndvi
        new_band_names = self.band_names + ['NDVI']

        return new_data, new_band_names

    def VHVV(self, data, v):
        band_vh = v[0]
        band_vv = v[1]

        vh = data[:, :, band_vh]
        vv = data[:, :, band_vv]

        mask = (vh == self.no_data_value) | (vv == self.no_data_value)

        vhvv = (vh - vv) / (vh + vv + 1e-10)
        vhvv[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = vhvv

        new_band_names = self.band_names + ['VHVV']

        return new_data, new_band_names

    def CRI(self, data, v):
        band_green = v[0]
        band_blue = v[1]

        green = data[:, :, band_green]
        blue = data[:, :, band_blue]

        mask = (green == self.no_data_value) | (blue == self.no_data_value)

        cri = green / (blue + 1e-10)
        cri[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = cri

        new_band_names = self.band_names + ['CRI']

        return new_data, new_band_names

    def BHDI1(self, data, v):
        band_re2 = v[0]
        band_blue = v[1]
        band_red = v[2]

        re2 = data[:, :, band_re2]
        blue = data[:, :, band_blue]
        red = data[:, :, band_red]

        mask = (re2 == self.no_data_value) | (blue == self.no_data_value) | (red == self.no_data_value)

        bhdi1 = re2 - blue + red
        bhdi1[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = bhdi1

        new_band_names = self.band_names + ['BHDI1']

        return new_data, new_band_names

    def BHDI2(self, data, v):
        band_re2 = v[0]
        band_red = v[1]
        band_re1 = v[2]
        band_blue = v[3]

        re2 = data[:, :, band_re2]
        red = data[:, :, band_red]
        re1 = data[:, :, band_re1]
        blue = data[:, :, band_blue]

        mask = (re2 == self.no_data_value) | (red == self.no_data_value) | (re1 == self.no_data_value) | (blue == self.no_data_value)

        bhdi2 = re2 + red + re1 - blue
        bhdi2[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = bhdi2

        new_band_names = self.band_names + ['BHDI2']

        return new_data, new_band_names

    def BHDI3(self, data, v):
        band_re1 = v[0]
        band_green = v[1]
        band_blue = v[2]
        band_red = v[3]

        re1 = data[:, :, band_re1]
        green = data[:, :, band_green]
        blue = data[:, :, band_blue]
        red = data[:, :, band_red]

        mask = (re1 == self.no_data_value) | (green == self.no_data_value) | (blue == self.no_data_value) | (red == self.no_data_value)

        bhdi3 = re1 + green - blue + red
        bhdi3[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = bhdi3

        new_band_names = self.band_names + ['BHDI3']

        return new_data, new_band_names

    def BHDI4(self, data, v):
        band_green = v[0]
        band_blue = v[1]
        band_red = v[2]

        green = data[:, :, band_green]
        blue = data[:, :, band_blue]
        red = data[:, :, band_red]

        mask = (green == self.no_data_value) | (blue == self.no_data_value) | (red == self.no_data_value)

        bhdi4 = 2 * green - blue - red
        bhdi4[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = bhdi4

        new_band_names = self.band_names + ['BHDI4']

        return new_data, new_band_names

    def REP(self, data, v):
        band_re3 = v[0]
        band_red = v[1]
        band_re2 = v[2]
        band_re1 = v[3]

        re3 = data[:, :, band_re3]
        red = data[:, :, band_red]
        re2 = data[:, :, band_re2]
        re1 = data[:, :, band_re1]

        mask = (re3 == self.no_data_value) | (red == self.no_data_value) | (re2 == self.no_data_value) | (re1 == self.no_data_value)

        rep = 705 + (35 * (0.5 * (re3 + red) - re1)) / (re2 - re1 + 1e-10)
        rep[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = rep

        new_band_names = self.band_names + ['REP']

        return new_data, new_band_names

    def WBI(self, data, v):
        idx = [band for band in v]
        bands_vals = [data[:, :, i] for i in idx]

        # Construct mask for any no_data_value in the used bands
        mask = np.zeros(bands_vals[0].shape, dtype=bool)
        for b in bands_vals:
            mask |= (b == self.no_data_value)

        numerator = bands_vals[0] + bands_vals[1] + bands_vals[2] + bands_vals[5]
        denominator = bands_vals[8] + bands_vals[9] + 1

        wbi = numerator / denominator
        wbi[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = wbi

        new_band_names = self.band_names + ['WBI']

        return new_data, new_band_names
    
    def GNDVI(self, data, v):
        band_nir = v[0]
        band_green = v[1]

        nir = data[:, :, band_nir]
        green = data[:, :, band_green]

        mask = (nir == self.no_data_value) | (green == self.no_data_value)

        gndvi = (nir - green) / (nir + green + 1e-10)
        gndvi[mask] = self.no_data_value
        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = gndvi
        new_band_names = self.band_names + ['GNDVI']

        return new_data, new_band_names

    def vis(self, data, index_name, bands_for_index, VIs):
        if index_name == 'CFI3':
            data, NDVI_TS = self.NDVI(data, VIs['NDVI'])
            data, features_vi = self.CFI3(data, bands_for_index, NDVI_TS)
        elif index_name == 'NDYI':
            data, features_vi = self.NDYI(data, bands_for_index)
        elif index_name == 'CRI':
            data, features_vi = self.CRI(data, bands_for_index)
        elif index_name == 'NDVI':
            data, features_vi = self.NDVI(data, bands_for_index)
        elif index_name == 'VHVV':
            data, features_vi = self.VHVV(data, bands_for_index)
        elif index_name == 'BHDI1':
            data, features_vi = self.BHDI1(data, bands_for_index)
        elif index_name == 'BHDI2':
            data, features_vi = self.BHDI2(data, bands_for_index)
        elif index_name == 'BHDI3':
            data, features_vi = self.BHDI3(data, bands_for_index)
        elif index_name == 'BHDI4':
            data, features_vi = self.BHDI4(data, bands_for_index)
        elif index_name == 'REP':
            data, features_vi = self.REP(data, bands_for_index)
        elif index_name == 'WBI':
            data, features_vi = self.WBI(data, bands_for_index)
        elif index_name == 'GNDVI':
            data, features_vi = self.GNDVI(data, bands_for_index)
        else:
            raise ValueError(f"Unknown index_name: {index_name}")

        return data, features_vi

    def make_VIs(self, data, VIs, features, feature_names_map):
        for index_name, bands_for_index in VIs.items():
            data, features_vi = self.vis(data, index_name, bands_for_index, VIs)
            features += features_vi
            feature_names_map[index_name] = index_name

        return data, features, feature_names_map


