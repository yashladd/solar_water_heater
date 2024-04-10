from pvlib.iotools import get_pvgis_hourly
import numpy as np
import json
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
from collections import namedtuple

# Define a namedtuple with all required parameters
SimulationParameters = namedtuple('SimulationParameters', [
    'A_c', 'F_r_tao_alpha', 'F_r_U_l',
    'm_l_dot', 'C_p', 'T_l', 'rho', 'consumption_pattern',
    'U_st', 'A_st', 'V_st', 'environment_conditions'
])



class Load:
    def __init__(self, config) -> None:
        # Desired hot water temperature during demand in °C
        self.T_l = config['desired_temperature']
        # Required load supply in litres
        self.litres_per_day = config['litres_per_day']
        # Consumption patter throughout the day
        self.consumption_pattern = config['consumption_pattern']
        # density kg/m^3
        self.rho = config['water_density']
        # Specific heat of the fluid we will consider only water J/kg°C
        self.C_p = config['specific_heat_water']
        # desired load mass flow rate, kg/s
        self.m_l_dot = self.calculate_mass_flow_rate()

    
    
    def calculate_mass_flow_rate(self):
        """
            Calculate the mass flow rate m_l_dot in kg/s
        """
        total_seconds = 0
    
        for period in self.consumption_pattern:
            start_hour, start_minute = map(int, period["start"].split(':'))
            end_hour, end_minute = map(int, period["end"].split(':'))
            
            start_time = start_hour * 3600 + start_minute * 60
            end_time = end_hour * 3600 + end_minute * 60
            
            # Calculate the duration of this period in seconds and add to the total
            duration = end_time - start_time
            total_seconds += duration
        total_kg_per_day = (self.litres_per_day / 1000) * self.rho
        return total_kg_per_day/total_seconds if total_seconds > 0 else 0



class Environment:
    def __init__(self, lat=18.53, lon=73.85, surface_tilt=0, tz='Asia/Kolkata') -> None:
        self.lat = lat
        self.lon = lon
        # TODO: Figure out angle now lets work with poa_direct from pvlib
        self.weather_conditions, _, _ = get_pvgis_hourly(
            latitude=lat, 
            longitude=lon,
            start=2006,
            end=2006,
            components=False,
            surface_tilt=surface_tilt
        )
        self.weather_conditions = self.weather_conditions.tz_convert(tz)
        self.weather_conditions.index = self.weather_conditions.index.floor('h')


class SolarCollector:
    def __init__(
            self, area, config
        ) -> None:
        #Colelctor area in m^2
        self.A_c = area
        # (collector heat removal factor) * (average transmittance absorptance product) => Fr(tao-alpha)
        self.F_r_tao_alpha = config['f_r_tao_alpha']
        # collector overall heat loss coefficient, W/m^2-°C
        self.F_r_U_l = config['f_r_u_l']
        # Tilt of the solar panel assumed to be facing south (azimuth=0)
        self.tilt = config['tilt']


class Storage:
    def __init__(self, volume, config) -> None:
        
        # Ration of height to diameter
        self.height_diameter_ratio = config['height_to_diameter_ratio']
        # Material of storage wall 
        self.wall_matetial = config['wall_material']
        # Thickness of the wall tank in m
        self.storage_wall_thicknes = config['storage_wall_thicknes']
        # Thermal conductivity of storage walls
        self.storage_wall_thermal_conductivity = config['thermal_conductivity_wall']
        # Insulation material used
        self.insulation_material = config['insulation_material']
        # Thickness of the insulation used
        self.insulation_thickness = config['insulation_thickness']
        # Thermal conductivity of insulation material 
        self.insulation_thermal_conductivity = config['insulation_thermal_conductivity']


        # Volume in m^3
        self.V_st = volume
        # storage heat loss coefficient, W/m^2°C
        self.U_st = self.estimate_storage_heat_loss_coefficient()
        # Surface area of the tank m^2
        self.A_st = self.calculate_surface_area()

    def calculate_surface_area(self):
        A_st = 1.845*(2 + self.height_diameter_ratio) * np.power(self.V_st, 2/3)
        return A_st
    
    def estimate_storage_heat_loss_coefficient(self):
        t_w = self.storage_wall_thicknes  # Thickness of the wall tank in m
        k_w = self.storage_wall_thermal_conductivity  # Thermal conductivity of storage walls in W/m-K
        t_ins = self.insulation_thickness  # Thickness of the insulation in m
        k_ins = self.insulation_thermal_conductivity  # Thermal conductivity of insulation material in W/m-K
        
        # Calculate the inverse of the sum of thermal resistances for the tank wall and the insulation
        # The internal and external heat transfer resistances are neglected, simplify the formula to:
        U_st = 1 / ((t_w / k_w) + (t_ins / k_ins))
        
        return U_st


class EnvironmentConditionsError(Exception):
    """Exception raised when fetching or processing environment conditions fails."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)



class SimulationEnvironment:
    # delta 
    _SIMULATION_STEP = 300
    _SECONDS_IN_A_DAY = 24 * 60 * 60
    _STEADY_STATE_STARTING_TEMPERATURE = 69.5 

    def __init__(self, collector_area, storage_tank_voume, config_file) -> None:

        with open(config_file) as f:
            config = json.load(f)

        self.environment = config['environment']
        self.simulation_year = config['simulation_year']
        self.solar_collector = SolarCollector(collector_area, config['solar_collector'])
        self.storage_tank = Storage(storage_tank_voume, config['storage_tank'])
        self.load = Load(config['load_profile'])
        self.environment_conditions = self.get_environment_conditions()
    
    def get_environment_conditions(self):
        try:
            data, _, _ = get_pvgis_hourly(
            latitude=self.environment['latitude'], 
            longitude=self.environment['longitude'],
            start=self.simulation_year,
            end=self.simulation_year,
            components=False,
            surface_tilt=self.solar_collector.tilt
        )
            data = data.tz_convert(self.environment['timezone'])
            # Floor the miutes to nearest hour for simulation
            data.index = data.index.floor('h')
            # Remove the timezone from the index for simulaiton 
            data.index = data.index.tz_localize(None)
            return data
        except Exception as e:
            raise EnvironmentConditionsError(f"Failed to fetch data from pvlib with these environment conditions:\n {e}")

    def create_simulation_parameters(self):
        """Create a SimulationParameters object encapsulating all necessary simulation parameters."""
        params = SimulationParameters(
            A_c=self.solar_collector.A_c,
            F_r_tao_alpha=self.solar_collector.F_r_tao_alpha,
            F_r_U_l=self.solar_collector.F_r_U_l,
            m_l_dot=self.load.m_l_dot,
            C_p=self.load.C_p,
            T_l=self.load.T_l,
            rho=self.load.rho,
            consumption_pattern=self.load.consumption_pattern,
            U_st=self.storage_tank.U_st,
            A_st=self.storage_tank.A_st,
            V_st=self.storage_tank.V_st,
            environment_conditions=self.environment_conditions
        )
        return params

    def simulate_single_day(self, month=4, day=15):
        simulation_start_datetime = datetime(self.simulation_year, month, day)
        simulation_period = self._SECONDS_IN_A_DAY

        self._run_sumilation(simulation_start_datetime, simulation_period)

    @staticmethod
    def get_solar_radiation_and_temperature(current_date_time, environment_conditions):
    
        # print(current_date_time)

        # Determine the hour in which current_date_time lies, This floors the time to the start of the current hour
        hour_start = current_date_time.replace(minute=0, second=0, microsecond=0)
        # print(hour_start)
        # Check if the hour_start exceeds the dataframe's range
        # If it does, use the last available hour in the dataframe
        if hour_start not in environment_conditions.index:
            hour_start = environment_conditions.index[-1]

        # Now, use hour_start to index the DataFrame and get poa_global and temp_air
        poa_global = environment_conditions.loc[hour_start, 'poa_global']
        temp_air = environment_conditions.loc[hour_start, 'temp_air']

        return poa_global, temp_air

    @staticmethod
    def is_water_consumed(current_time, water_consumption):
        is_water_consumed = False
        
        for period in water_consumption:
            # Convert start and end times to datetime.time for comparison
            start_time = datetime.strptime(period["start"], "%H:%M").time()
            end_time = datetime.strptime(period["end"], "%H:%M").time()
            
            # Check if current_time falls within the consumption period
            if start_time <= current_time < end_time:
                # print("REACHED")
                is_water_consumed = True
                break  # Exit the loop if a matching period is found
        return is_water_consumed
    
    @staticmethod
    def handle_case_1(params, I_t, T_sti, T_a, delta_t):
        """
            Solving equation 11a analytically
        """
        k_1 = params.A_c * params.F_r_U_l + params.U_st * params.A_st
        k_2 = params.A_c * I_t * params.F_r_tao_alpha + params.A_c * params.F_r_U_l * T_a - params.m_l_dot * params.C_p * (params.T_l - T_a) + params.U_st * params.A_st * T_a
        k_3 = ((params.U_st * params.A_st + params.A_c * params.F_r_U_l) / (params.rho * params.C_p * params.V_st))
        T_stf = (k_2 - (k_2 - k_1 * T_sti) * math.exp(-k_3 * delta_t)) / k_1
        return T_stf

    @staticmethod
    def handle_case_2(params, I_t, T_sti, T_a, delta_t):
        """
            Solving equation 11b analytically
        """
        # c1
        heat_supplied_load = params.m_l_dot * params.C_p * (params.T_l - T_a)
        # Storage coeff c2
        storage_loss_coeff = params.U_st * params.A_st
        # c_3
        c_3 = ((params.U_st * params.A_st) / (params.rho * params.C_p * params.V_st))
        T_stf = (storage_loss_coeff * T_a - heat_supplied_load 
                - math.exp(-c_3 * delta_t) * (storage_loss_coeff * T_a - heat_supplied_load - storage_loss_coeff * T_sti)) / (storage_loss_coeff)
        return T_stf

    @staticmethod
    def handle_case_3(params, I_t, T_sti, T_a, delta_t):
        """
            Solving equation 11c analytically
        """
        # k1
        k_1 = params.A_c * I_t * params.F_r_tao_alpha + T_a * (params.A_c * params.F_r_U_l +params.m_l_dot * params.C_p + params.U_st * params.A_st)
        # k_2
        k_2 = params.A_c * params.F_r_U_l + params.m_l_dot * params.C_p + params.U_st * params.A_st
        #k_3
        k_3 = (params.A_c * params.F_r_U_l + params.U_st * params.A_st + params.m_l_dot * params.C_p) / (params.rho * params.C_p * params.V_st)

        T_stf = (k_1 - math.exp(- k_3 * delta_t) * (k_1 - k_2 * T_sti)) / k_2
        return T_stf

    @staticmethod
    def handle_case_4(params,I_t, T_sti, T_a, delta_t):
        """
            Solving equation 11d analytically
        """
        c_1 = params.m_l_dot * params.C_p
        c_2 = params.U_st * params.A_st
        c_3 = (c_1 + c_2) / (params.rho * params.C_p * params.V_st)

        T_stf = (T_a * (c_1 + c_2) - math.exp(-(c_3 * delta_t)) * (T_a * (c_1 + c_2) - T_sti * (c_1 + c_2))) / (c_1 + c_2)
        return T_stf

    @staticmethod
    def get_solar_useful_gain_rate(params, I_t, T_st, T_a):
        if int(I_t) == 0:
            return 0.0
        return params.A_c * (I_t * params.F_r_tao_alpha - params.F_r_U_l * (T_st - T_a))

    # @staticmethod
    def get_next_temperature(self, params, I_t, T_sti, T_l, T_a, delta_t):
        """
            Solve the differential equations from the energy balance of the tank
        """
        q_s = self.get_solar_useful_gain_rate(params, I_t, T_sti, T_a)
        # q_s = q_s if q_s > 0 else 0.0

        # Case 1
        if T_sti > T_l and q_s > 0:
            return self.handle_case_1(params, I_t, T_sti, T_a, delta_t)
        elif T_sti > T_l and q_s <= 0:
            return self.handle_case_2(params, I_t, T_sti, T_a, delta_t)
        elif T_sti <= T_l and q_s > 0:
            return self.handle_case_3(params, I_t, T_sti, T_a, delta_t)
        else:
            return self.handle_case_4(params, I_t, T_sti, T_a, delta_t)

    def _run_sumilation(self,  start_datetime, period):

        simulation_params = self.create_simulation_parameters()

        # Initialize steady state starting temperature
        temperature_profile = [self._STEADY_STATE_STARTING_TEMPERATURE]
        delta_t = self._SIMULATION_STEP
        
        for timestep in range(0, period, delta_t):
            # Calculate the current simulation time
            current_date_time = start_datetime + timedelta(seconds=timestep)
            

            I_t, T_a = self.get_solar_radiation_and_temperature(current_date_time, simulation_params.environment_conditions)
            # print(I_t, T_a)

            current_time = current_date_time.time()
            is_consumed = self.is_water_consumed(current_time, simulation_params.consumption_pattern)
            # print(is_water_consumed(current_time, water_consumption))
            print("CRRTIME", current_time)
            if not is_consumed:
                temperature_profile.append(temperature_profile[-1])
            else:
                temperature_profile.append(self.get_next_temperature(simulation_params, I_t, temperature_profile[-1], simulation_params.T_l, T_a, delta_t))
        print(len(temperature_profile))


        # Re-importing necessary libraries and re-plotting after reset


        # Plotting the array
        plt.figure(figsize=(10, 6))
        plt.plot(temperature_profile, color='blue', label='Array Values')

        # Plotting a line parallel to the x-axis at y=60
        plt.axhline(y=60, color='red', linestyle='--', label='Line at y=60')

        plt.ylim(20, max(temperature_profile) + 30)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Plot with a Line Parallel to the X-axis at y=60')
        plt.legend()
        plt.grid(True)
        plt.show()