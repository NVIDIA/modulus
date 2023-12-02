# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cdsapi
import numpy as np
import os

usr = 'j' # j,k,a
base_path = '/project/projectdirs/dasrepo/ERA5/wind_levels/' + usr
if not os.path.isdir(base_path):
    os.makedirs(base_path)
year_dict = {'j': np.arange(1979, 1993), 'k': np.arange(1993, 2006), 'a' : np.arange(2006, 2021)}
years = year_dict[usr]  
#t1 = [str(jj).zfill(2) for jj in range(1,4)] 
#t2 = [str(jj).zfill(2) for jj in range(4,7)] 
#t3 = [str(jj).zfill(2) for jj in range(7,10)] 
#t4 = [str(jj).zfill(2) for jj in range(10,13)] 
#
#trimesters = [t1, t2, t3, t4]
months = [str(jj).zfill(2) for jj in range(1,13)] 
 
pressure_levels = [1000, 900, 800, 700]
c = cdsapi.Client()

for pressure_level in pressure_levels:
    
    for year in years:

      for month in months:
        
        print(month)
        year_str = str(year) 
        pressure_str = str(pressure_level)
        month_str = month 
        file_str = base_path + '/u_v_pressure_level_'+ pressure_str + '_'  + year_str + '_month_' + month_str + '.nc'
        print(year_str)
        print(file_str)
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'pressure_level': pressure_str,
                'variable': [
                    'u_component_of_wind', 'v_component_of_wind',
                ],          
                'year': year_str,
                'month': month_str,
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],          },
            file_str)
