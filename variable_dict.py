variable_list = ['uv_storm_194', 'uv_storm_195', 'Specific_humidity_500', 'Specific_humidity_850', \
                 'temperature_1000mbar', 'Absolute_vorticity_500', 'Absolute_vorticity_850', \
                 'Uwind_500', 'Uwind_850', 'Vwind_500', 'Vwind_850']

'''建立变量和最大最小值匹配字典'''
variable_dict_max = {'Absolute_vorticity_500':0.0016, 'Absolute_vorticity_700':0.0018,\
                     'Absolute_vorticity_850':0.0019, 'Specific_humidity_500':0.0114,\
                     'Specific_humidity_700':0.0218, 'Specific_humidity_850':0.0338,\
                     'temperature_1000mbar':328.00, 'uv_storm_194':61.80,\
                     'uv_storm_195':66.20, 'Uwind_500':82.80,
                     'Uwind_700':64.80, 'Uwind_850':64.40,\
                     'Vwind_500':77.30, 'Vwind_700':70.70,\
                     'Vwind_850':67.72, 'vertical_speed_tropopause':0.0790}
variable_dict_min = {'Absolute_vorticity_500':'nan', 'Absolute_vorticity_700':'nan',\
                     'Absolute_vorticity_850':'nan', 'Specific_humidity_500':0,\
                     'Specific_humidity_700':0, 'Specific_humidity_850':0,\
                     'temperature_1000mbar':210.00, 'uv_storm_194':'nan',\
                     'uv_storm_195':'nan', 'Uwind_500':'nan',
                     'Uwind_700':'nan', 'Uwind_850':'nan',\
                     'Vwind_500':'nan', 'Vwind_700':'nan',\
                     'Vwind_850':'nan', 'vertical_speed_tropopause':'nan'}
