from cis.data_io.products.AProduct import AProduct


class spexone_L2(AProduct):
    
    priority = 199
    
    def get_file_signature(self):
        return [r'PACE_SPEXONE.*REMOTAP.nc']
    
    def _read_coord_data(self, filenames):
        """Helper function to read coordinate data from files
        Returns concatenated coordinates"""
        from cis.data_io.netcdf import read_many_files_individually
        from cis.time_util import cis_standard_time_unit
        from datetime import datetime, timedelta
        
        import numpy as np
        
        variables = [
        "geolocation_data/longitude",
        "geolocation_data/latitude", 
        "geolocation_data/fracday",
        "geolocation_data/utc_date"
        ]
        
        # Read the data with the fully qualified variable names
        data = read_many_files_individually(filenames, variables)

        # Concatenate and flatten the data for each variable
        flat_lons = np.concatenate([d[:].flatten() for d in data["geolocation_data/longitude"]])
        flat_lats = np.concatenate([d[:].flatten() for d in data["geolocation_data/latitude"]])
        flat_fracday = np.concatenate([d[:].flatten() for d in data["geolocation_data/fracday"]])
        flat_dates = np.concatenate([d[:].flatten() for d in data["geolocation_data/utc_date"]])

        # Format reference dates for each point
        ref_dates = [f"{int(date):.0f}"[:4] + "-" + f"{int(date):.0f}"[4:6] + "-" + f"{int(date):.0f}"[6:8] for date in flat_dates]

        # Convert to datetime objects
        datetimes = []
        for date_str, frac in zip(ref_dates, flat_fracday):
            date = datetime.strptime(date_str, '%Y-%m-%d')
            time_delta = timedelta(days=float(frac))
            datetimes.append(date + time_delta)
            
        # Convert to standard time using cis_standard_time_unit
        standard_time = cis_standard_time_unit.date2num(datetimes)

        return flat_lons, flat_lats, standard_time
        
    def _create_coords(self, filenames, lons, lats, times):
        from cis.data_io.Coord import Coord, CoordList
        from cis.data_io.netcdf import get_metadata, read
        from cis.data_io.ungridded_data import Metadata
        import numpy as np

        variables = [
            "geolocation_data/longitude",
            "geolocation_data/latitude",
        ]
        
        # Sort by time
        sort_indices = np.argsort(times)
        times = times[sort_indices]
        lats = lats[sort_indices]
        lons = lons[sort_indices]
        
        # Create metadata and coordinates
        data = read(filenames[0], variables)        
        lon = Coord(lons, get_metadata(data["geolocation_data/longitude"]), "X")
        lat = Coord(lats, get_metadata(data["geolocation_data/latitude"]), "Y")
        time = Coord(times, Metadata(name="Time", shape=(len(times),),
                                    units="days since 1600-01-01 00:00:00",
                                    standard_name="time", long_name="time"), "T")
        
        coords = CoordList([lat, lon, time])
        return coords, sort_indices
    
    def create_coords(self, filenames):
        from cis.data_io.ungridded_data import UngriddedCoordinates
        lons, lats, times = self._read_coord_data(filenames)
        coords, _ = self._create_coords(filenames, lons, lats, times)
        return UngriddedCoordinates(coords)
    
    def get_nwave_ind(self, filename, wave_length, group="geophysical_data"):
        from cis.data_io.netcdf import read
        import numpy as np
        
        data = read(filename, [f"{group}/wave_optic_prop"])
        wavelengths = data[f"{group}/wave_optic_prop"][:]
        
        nwave_ind = (np.abs(wavelengths - wave_length)).argmin()
        
        return nwave_ind
    
    # List of variables that have wavelength dimension
    wavelength_vars = [
        "aot", "aot_uncertainty", "aot_fine", "aot_fine_uncertainty", 
        "aot_coarse", "aot_coarse_uncertainty", "ssa", "ssa_uncertainty", 
        "ssa_fine", "ssa_fine_uncertainty", "ssa_coarse", "ssa_coarse_uncertainty", 
        "lidar_p11_pi", "lidar_bsca_total", "lidar_depol_ratio", "lidar_ratio", 
        "mr_fine", "mr_coarse", "mr", "mi_fine", "mi_coarse", "mi", "fmf", 
        "mr_mode1", "mr_mode1_uncertainty", "mr_mode2", "mr_mode2_uncertainty", 
        "mr_mode3", "mr_mode3_uncertainty", "mi_mode1", "mi_mode1_uncertainty", 
        "mi_mode2", "mi_mode2_uncertainty", "mi_mode3", "mi_mode3_uncertainty", 
        "aot_mode1", "aot_mode1_uncertainty", "aot_mode2", "aot_mode2_uncertainty", 
        "aot_mode3", "aot_mode3_uncertainty", "ssa_mode1", "ssa_mode1_uncertainty", 
        "ssa_mode2", "ssa_mode2_uncertainty", "ssa_mode3", "ssa_mode3_uncertainty"
    ]
        
        
    def _read_variable_data(self, filenames, variable):
        from cis.data_io.netcdf import read
        from cis.utils import concatenate
        import re
        import numpy as np
        
        # Check if variable includes group specification (format: group/variable)
        if '/' in variable:
            group, var = variable.split('/', 1)
        else:
            group = "geophysical_data"
            var = variable
        
        # Check if this is a wavelength-specific variable
        wave_match = re.search(r'(.+)_(\d+)nm$', var)
        
        if wave_match:
            base_var = wave_match.group(1)
            wavelength = float(wave_match.group(2))
            
            full_variable = f"{group}/{base_var}"
            
            if base_var not in self.wavelength_vars:
                raise ValueError(f"Variable {base_var} does not have wavelength dimension")
                
            all_data = []
            for filename in filenames:
                nwave_ind = self.get_nwave_ind(filename, wavelength)
                data = read(filename, [full_variable])
                # Flatten 2D data and select wavelength index
                all_data.append(data[full_variable][:, :, nwave_ind].flatten())
            
            return concatenate(all_data), f"{base_var} at {wavelength}nm"
        else:
            full_variable = f"{group}/{var}"
            all_data = []
            for filename in filenames:
                data = read(filename, [full_variable])
                # Flatten 2D data
                all_data.append(data[full_variable][:].flatten())
            
            return concatenate(all_data), var
    
    def create_data_object(self, filenames, variable):
        from cis.data_io.ungridded_data import UngriddedData, Metadata
        from cis.data_io.netcdf import get_metadata, read
        import re
        
        # Read raw data
        data, long_name = self._read_variable_data(filenames, variable)
        lons, lats, times = self._read_coord_data(filenames)
        
        # Create coordinates
        coords, sort_indices = self._create_coords(filenames, lons, lats, times)
        
        # Sort the data according to time
        data = data[sort_indices]
        
        '''
        wave_match = re.search(r'(.+)_(\d+)nm$', variable)
        if wave_match and variable in self.wavelength_vars:
            base_var = wave_match.group(1)
            wavelength = float(wave_match.group(2))
            
            if base_var not in self.wavelength_vars:
                raise ValueError(f"Variable {base_var} does not have wavelength dimension")
        
            metadata = get_metadata(read(filenames[0], ["geophysical_data/" + base_var]))
            
            metadata.long_name = f"{metadata.long_name} at {wavelength}nm"
        else:
            pass
            metadata = get_metadata(read(filenames[0], ["geophysical_data/" + variable]))
        '''
        
        metadata = Metadata(name=variable,
                            shape=(len(data),),
                            long_name=long_name,
                            range=(data.min(), data.max()))
        # Create the ungridded data object
        return UngriddedData(data, metadata, coords)
        

        
class spexone_gridded(AProduct):

    priority = 199
    
    def get_file_signature(self):
        # We don't know of any 'standard' netCDF CF model data yet...
        return [r'l2_gridded_\d{8}\\.nc']
    
    def _read_coord_data(self, filenames):
        """Helper function to read coordinate data from files
        Returns concatenated coordinates"""
        from cis.data_io.netcdf import read
        from cis.utils import concatenate
        import re
        
        all_lats = []
        all_lons = []
        all_times = []
        
        # Process each file individually
        for filename in sorted(filenames):  # Sort filenames for consistency
            # Read data for this file
            data = read(filename, ["longitude", "latitude", "fracday"])
            
            # Extract reference date from filename
            match = re.search(r'l2_gridded_(\d{8})\.nc$', filename)
            if match:
                date_str = match.group(1)
                ref_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            else:
                raise ValueError(f"Could not extract date from filename: {filename}")
            
            # Convert fracday to standard time for this file
            from cis.time_util import convert_time_using_time_stamp_info_to_std_time
            converted_time = convert_time_using_time_stamp_info_to_std_time(
                data["fracday"][:], 
                "days", 
                ref_date
            )
            
            all_lons.append(data["longitude"][:])
            all_lats.append(data["latitude"][:])
            all_times.append(converted_time)
        
        # Concatenate all coordinates
        concat_lons = concatenate(all_lons)
        concat_lats = concatenate(all_lats)
        concat_times = concatenate(all_times)
        return concat_lons, concat_lats, concat_times

    def _create_coords(self, lons, lats, times):
        """Helper function to create and sort coordinates
        Returns coords and sort indices for data alignment"""
        from cis.data_io.Coord import Coord, CoordList
        from cis.data_io.ungridded_data import Metadata
        import numpy as     np
        
        # Sort by time
        sort_indices = np.argsort(times)
        times = times[sort_indices]
        lats = lats[sort_indices]
        lons = lons[sort_indices]
        
        # Create metadata and coordinates
        lon_metadata = Metadata(name="Longitude", shape=(len(lons),), 
                              units="degrees_east", standard_name="longitude",
                              long_name="longitude",
                              range=(lons.min(), lons.max()))
        lon = Coord(lons, lon_metadata, "X")
        
        lat_metadata = Metadata(name="Latitude", shape=(len(lats),),
                              units="degrees_north", standard_name="latitude",
                              long_name="latitude",
                              range=(lats.min(), lats.max()))
        lat = Coord(lats, lat_metadata, "Y")
        
        time_metadata = Metadata(name="Time", shape=(len(times),),
                               units="days since 1600-01-01 00:00:00",
                               standard_name="time", long_name="time")
        time = Coord(times, time_metadata, "T")
        
        coords = CoordList([lat, lon, time])
        return coords, sort_indices

    def create_coords(self, filenames):
        from cis.data_io.ungridded_data import UngriddedCoordinates
        lons, lats, times = self._read_coord_data(filenames)
        coords, _ = self._create_coords(lons, lats, times)
        return UngriddedCoordinates(coords)
    
    def get_nwave_ind(self, filename, wave_length):
        from cis.data_io.netcdf import read
        import numpy as np
        
        data = read(filename, ["wavelength"])
        wavelengths = data["wavelength"][:]
        nwave_ind = (np.abs(wavelengths - wave_length)).argmin()
        
        return nwave_ind
    
    # List of variables that have wavelength dimension
    wavelength_vars = [
        'aot', 'ssa', 'aot_fine', 'aot_coarse', 'mr', 'mi',
        'aot_uncertainty', 'ssa_uncertainty', 'aot_fine_uncertainty', 
        'aot_coarse_uncertainty'
    ]

    def _read_variable_data(self, filenames, variable):
        """Helper function to read and concatenate variable data"""
        from cis.data_io.netcdf import read
        from cis.utils import concatenate
        import re
        import numpy as np
        
        # Handle aaod_{wl}nm
        aaod_match = re.search(r'aaod_(\d+)nm$', variable)
        if aaod_match:
            wavelength = float(aaod_match.group(1))
            
            all_aot = []
            all_ssa = []
            for filename in filenames:
                nwave_ind = self.get_nwave_ind(filename, wavelength)
                data = read(filename, ['aot', 'ssa'])
                all_aot.append(data['aot'][:, nwave_ind])
                all_ssa.append(data['ssa'][:, nwave_ind])
            
            concat_aot = concatenate(all_aot)
            concat_ssa = concatenate(all_ssa)
            concat_aaod = concat_aot * (1 - concat_ssa)
            
            return concat_aaod, f"aaod at {wavelength}nm"
        
        # Handle angstrom_{wl1}nm_{wl2}nm
        angstrom_match = re.search(r'angstrom_(\d+)nm_(\d+)nm$', variable)
        if angstrom_match:
            wl1 = float(angstrom_match.group(1))
            wl2 = float(angstrom_match.group(2))
            
            all_aot_wl1 = []
            all_aot_wl2 = []
            for filename in filenames:
                nwave_ind_wl1 = self.get_nwave_ind(filename, wl1)
                nwave_ind_wl2 = self.get_nwave_ind(filename, wl2)
                data = read(filename, ['aot'])
                all_aot_wl1.append(data['aot'][:, nwave_ind_wl1])
                all_aot_wl2.append(data['aot'][:, nwave_ind_wl2])
            
            concat_aot_wl1 = concatenate(all_aot_wl1)
            concat_aot_wl2 = concatenate(all_aot_wl2)
            
            angstrom = -np.log(concat_aot_wl1 / concat_aot_wl2) / np.log(wl1 / wl2)
            
            return angstrom, f"Angstrom exponent between {wl1}nm and {wl2}nm"
        
        # Check if this is a wavelength-specific request
        wave_match = re.search(r'(.+)_(\d+)nm$', variable)
        if wave_match:
            base_var = wave_match.group(1)
            wavelength = float(wave_match.group(2))
            
            if base_var not in self.wavelength_vars:
                raise ValueError(f"Variable {base_var} does not have wavelength dimension")
                
            all_data = []
            for filename in filenames:
                nwave_ind = self.get_nwave_ind(filename, wavelength)
                data = read(filename, [base_var])
                all_data.append(data[base_var][:, nwave_ind])
            
            return concatenate(all_data), f"{base_var} at {wavelength}nm"
        else:
            all_data = []
            for filename in filenames:
                data = read(filename, [variable])
                all_data.append(data[variable][:])
            return concatenate(all_data), variable

    def _clean_data(self, data, lons, lats, times):
        """Remove invalid data and update coordinates accordingly"""
        import numpy as np
        
        # Create masked arrays
        masked_data = np.ma.masked_invalid(data)
        masked_lons = np.ma.masked_invalid(lons)
        masked_lats = np.ma.masked_invalid(lats)
        masked_times = np.ma.masked_invalid(times)
        
        # Combine all masks
        combined_mask = (masked_data.mask | 
                        masked_lons.mask | 
                        masked_lats.mask | 
                        masked_times.mask)
        
        # Remove invalid data points
        valid_data = masked_data.data[~combined_mask]
        valid_lons = masked_lons.data[~combined_mask]
        valid_lats = masked_lats.data[~combined_mask]
        valid_times = masked_times.data[~combined_mask]
        
        return valid_data, valid_lons, valid_lats, valid_times

    def create_data_object(self, filenames, variable):
        from cis.data_io.ungridded_data import UngriddedData, Metadata
        import numpy as np
        
        fill_value = -32767
        
        # Read raw data
        data, long_name = self._read_variable_data(filenames, variable)
        lons, lats, times = self._read_coord_data(filenames)
        
        # Create coordinates
        coords, sort_indices = self._create_coords(lons, lats, times)
        
        # Sort the data according to time
        data = data[sort_indices]
        
        # Create mask for NaN values
        nan_mask = np.isnan(data)
        
        # Replace NaN values with fill_value and create masked array
        data = np.where(nan_mask, fill_value, data)
        masked_data = np.ma.masked_array(data, mask=nan_mask)
        
        # Create metadata
        metadata = Metadata(name=variable, 
                          missing_value=fill_value,
                          shape=(len(data),),
                          long_name=long_name,
                          range=(masked_data.min(), masked_data.max()),
        )
        
        # Create the ungridded data object with masked data
        return UngriddedData(masked_data, metadata, coords)
