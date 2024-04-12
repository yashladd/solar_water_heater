from matplotlib.lines import Line2D
from pvlib.iotools import get_pvgis_hourly
import numpy as np
import json
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
from collections import namedtuple
import matplotlib.dates as mdates
import calendar
import pandas as pd


# Define a namedtuple with all required parameters
SimulationParameters = namedtuple('SimulationParameters', [
    'A_c', 'V_st', 'F_r_tao_alpha', 'F_r_U_l',
    'm_l_dot', 'C_p', 'T_l', 'rho', 'consumption_pattern',
    'U_st', 'A_st',  'environment_conditions'
])



class Load:
    """
        Defines the load properties, consumption pattern for the scenario and calculates the mass flow rate
    """
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


class SolarCollector:
    """
        Models the properties of solar collector 
    """
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
    """
        Models the & calculates the properties of the storage tank based its charachteristics
    """
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
    """
        Main logic of simulating The temerature of the stroage water by varying:
            1. Solar Collector Area
            2. Storage Tank Volume

        Creates Simulation for:
            1. Single Day 
            2. Entire Month
            3. Entire Year
    """    
    _SIMULATION_STEP = 300
    _SECONDS_IN_A_DAY = 24 * 60 * 60
    _STEADY_STATE_STARTING_TEMPERATURE = 69.5 

    def __init__(self, collector_area, storage_tank_volume, config_file) -> None:
        self.config_file = config_file

        with open(self.config_file) as f:
            config = json.load(f)

        self._collector_area = collector_area
        self._storage_tank_volume = storage_tank_volume

        self.environment = config['environment']
        self.simulation_year = config['simulation_year']
        self.solar_collector = SolarCollector(collector_area, config['solar_collector'])
        self.storage_tank = Storage(storage_tank_volume, config['storage_tank'])
        self.load = Load(config['load_profile'])
        self.environment_conditions = self.get_environment_conditions()
        self.simulation_params = self.create_simulation_parameters()

    @property
    def collector_area(self):
        return self._collector_area

    @collector_area.setter
    def collector_area(self, value):
        self._collector_area = value
        self.simulation_params = self.create_simulation_parameters()

    @property
    def storage_tank_volume(self):
        return self._storage_tank_volume

    @storage_tank_volume.setter
    def storage_tank_volume(self, value):
        self._storage_tank_volume = value
        self.simulation_params = self.create_simulation_parameters()

    def get_environment_conditions(self):
        """
            Get the Solar radiation and ambient temperature for the current location according to config_file provided
        """
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
        # print("Updated params")
        params = SimulationParameters(
            A_c=self._collector_area,
            V_st=self._storage_tank_volume,
            F_r_tao_alpha=self.solar_collector.F_r_tao_alpha,
            F_r_U_l=self.solar_collector.F_r_U_l,
            m_l_dot=self.load.m_l_dot,
            C_p=self.load.C_p,
            T_l=self.load.T_l,
            rho=self.load.rho,
            consumption_pattern=self.load.consumption_pattern,
            U_st=self.storage_tank.U_st,
            A_st=self.storage_tank.A_st,
            environment_conditions=self.environment_conditions
        )
        return params


    @staticmethod
    def get_solar_radiation_and_temperature(current_date_time, environment_conditions):
        """
            Get I_t and T_a at the simulation instant
        """
        # print(current_date_time)

        # Determine the hour in which current_date_time lies, This floors the time to the start of the current hour
        hour_start = current_date_time.replace(minute=0, second=0, microsecond=0)
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
        
        k_2 = (
                params.A_c * I_t * params.F_r_tao_alpha + params.A_c * params.F_r_U_l * T_a 
                - params.m_l_dot * params.C_p * (params.T_l - T_a) 
                + params.U_st * params.A_st * T_a
            )
        
        k_3 = (params.U_st * params.A_st + params.A_c * params.F_r_U_l) / (params.rho * params.C_p * params.V_st)

        T_stf = (k_2 - (k_2 - k_1 * T_sti) * math.exp(-k_3 * delta_t)) / k_1
        return T_stf

    @staticmethod
    def handle_case_2(params, I_t, T_sti, T_a, delta_t):
        """
            Solving equation 11b analytically
        """
        # c1
        c_1 = params.m_l_dot * params.C_p * (params.T_l - T_a)
        # Storage coeff c2
        c_2 = params.U_st * params.A_st
        # c_3
        c_3 = ((params.U_st * params.A_st) / (params.rho * params.C_p * params.V_st))
        T_stf = (c_2 * T_a - c_1 
                - math.exp(-c_3 * delta_t) * (c_2 * T_a - c_1 - c_2 * T_sti)) / (c_2)
        return T_stf

    @staticmethod
    def handle_case_3(params, I_t, T_sti, T_a, delta_t):
        """
            Solving equation 11c analytically
        """
        # k_0
        k_0 = (params.A_c * params.F_r_U_l + params.m_l_dot * params.C_p + params.U_st * params.A_st)
        
        # k_1
        k_1 = ( params.A_c * I_t * params.F_r_tao_alpha + T_a * k_0)

        #k_2
        k_2 = (k_0) / (params.rho * params.C_p * params.V_st)

        T_stf = (k_1 - math.exp(-k_2 * delta_t) * (k_1 - k_0 * T_sti)) / k_0
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
    
    @staticmethod
    def calculate_total_energy_rate(params, T_a):
        m_l_dot = params.m_l_dot
        C_p = params.C_p
        T_l = params.T_l
        return m_l_dot * C_p * (T_l - T_a) 
    
    @staticmethod
    def calculate_energy_rate_met_by_auxiliary(params, T_st, T_a):
        m_l_dot = params.m_l_dot
        C_p = params.C_p
        T_l = params.T_l
        # print("AUX_CALC" + " T_ST T_l", T_st, T_a)
        # Total Energy will be met by Solar 
        if T_st > T_l: return 0
        # Total energy requirement
        total_energy_rate = m_l_dot * C_p * (T_l - T_a) 
        # No energy can be met by solar total supplied by auxiliary
        if T_st <= T_a: return total_energy_rate
        solar_energy_rate = m_l_dot * C_p * (T_st - T_a) 
        return total_energy_rate - solar_energy_rate
    
    def plot_water_consumption(self):
        """
            Plot the water consumption pattern of the environment
        """
        load_profiles = self.simulation_params.consumption_pattern
        hours = np.arange(0, 24, 1)
        load_presence = np.zeros_like(hours)
        def time_to_hour(time_str):
            hour, minute = map(int, time_str.split(':'))
            return hour + minute / 60
        for profile in load_profiles:
            start_hour = time_to_hour(profile["start"])
            end_hour = time_to_hour(profile["end"])
            # Using boolean indexing to set the load presence to 1 for the appropriate hours
            load_presence[(hours >= start_hour) & (hours < end_hour)] = 1
        plt.figure(figsize=(12, 3))
        plt.step(hours, load_presence, where='post', linewidth=2, color='blue')

        plt.gca().spines['left'].set_position(('data', 0))
        plt.gca().spines['bottom'].set_position(('data', 0))
        plt.gca().spines['right'].set_color('none')
        plt.gca().spines['top'].set_color('none')

        plt.yticks([1])
        plt.xticks(hours)
        plt.xlabel('Time of the day')
        plt.ylabel('Hot water consumption')
        plt.title('Load Profile Over 24 Hours')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_solar_radiation(df, month=None, day=None):
        """
        Plot the poa_global against time from a pandas DataFrame.
        
        :param df: pandas DataFrame with a DateTimeIndex and 'poa_global' column.
        :param month: int or str, optional month number (1-12) or name.
        :param day: int, optional day of the month.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df[df.index.year <= df.index.year[0]]
        plt.figure(figsize=(12, 6))
        # Filter data based on the month and day provided
        if month is not None and day is not None:
            # Specific month and day
            start_date = f"{df.index.year[0]}-{month:02d}-{day:02d}"
            end_date = f"{df.index.year[0]}-{month:02d}-{day:02d} 23:59:59"
            df_filtered = df[start_date:end_date]
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
            plt.gca().xaxis.set_major_locator(mdates.HourLocator())
        elif month is not None:
            # Specific month
            df_filtered = df[df.index.month == int(month)]
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d'))  # Show day of the month
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
        else:
            # Entire year
            df_filtered = df
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        print(df_filtered.index)
        # Plotting
        plt.plot(df_filtered.index, df_filtered['poa_global'], label='poa_global')
        plt.xlabel('Time')
        plt.ylabel('Solar Radiation on tilted surface I_t')
        plt.title('Solar radiation Over Time')
        plt.xlim(df_filtered.index[0], df_filtered.index[-1])
        plt.ylim(bottom=0)
        # plt.legend()
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_energy_bar_graph(formatted_total_energy, formatted_energy_auxiliary):
        """
        Plot a bar graph of the total energy required and the energy met by auxiliary.
        Also, display the solar fraction as a legend.

        :param total_energy: float, total energy required.
        :param energy_auxiliary: float, energy met by auxiliary.
        """
        # Calculate solar fraction
        # Extract numerical values for plotting
        total_energy_value = float(formatted_total_energy.split()[0])
        energy_auxiliary_value = float(formatted_energy_auxiliary.split()[0])
        solar_fraction = 1 - (energy_auxiliary_value / total_energy_value)
        labels = ['Total Energy Required', 'Energy Met by Auxiliary']
        energies = [total_energy_value, energy_auxiliary_value]

        fig, ax = plt.subplots()
        ax.bar(labels[0], energies[0], color='blue')
        ax.bar(labels[1], energies[1], color='red')
        ax.set_ylabel(f'Energy ({formatted_total_energy.split()[1]})')
        ax.set_title('Total vs Auxiliary Energy')
        # ax.legend([f'Solar Fraction: {solar_fraction:.3f}'])
        legend_handle = Line2D([0], [0], linestyle='none', marker='s', markerfacecolor='yellow', markeredgewidth=0.0)

        # Add legend to the plot with custom handle for the solar fraction
        ax.legend(handles=[legend_handle], labels=[f'Solar Fraction: {solar_fraction:.3f}'], handlelength=1, handletextpad=0.5)


        # Show the plot
        plt.show()

        # # Create bar graph
        # plt.figure(figsize=(10, 6))
        # bar_width = 0.35
        # index = [0, 1]
        # energies = [total_energy_value, energy_auxiliary_value]
        # labels = ['Total Energy Required', 'Energy Met by Auxiliary']

        # plt.bar(index, energies, bar_width, label=f'Solar Fraction: {solar_fraction:.3f}')

        # # Add labels, title, and legend
        # plt.xlabel('Category')
        # plt.ylabel('Energy (' + formatted_total_energy.split()[1] + ')')
        # plt.title('Energy Comparison with Solar Fraction')
        # plt.xticks(index, labels)
        # plt.legend()

        # # Show plot
        # plt.tight_layout()
        # plt.show()


    @staticmethod
    def plot_single_day(storage_temperature, date_times):
        """
            Plots the temeperature profile for single day simulation 
        """
        print("DATE_TIME_LEN", len(date_times))
        print("STORAGE_WATER_TEMPERATURE_LEN", len(storage_temperature))
        plt.figure(figsize=(12, 6))
        plt.plot(date_times, storage_temperature, color='blue', label='Storage Water Temperature')

        # Plotting a line parallel to the x-axis at y=60
        plt.axhline(y=60, color='red', linestyle='--', label='Load Temperature requirement = 60°C')
        plt.axhline(y=100, color='red', linestyle='--', label='Limiting Temerature Line = 100°C')

        # Formatting the x-axis to show hours as integers
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator())

        # Explicitly set the limits for x-axis to show from 0 to 24
        plt.xlim(date_times[0], date_times[-1] + timedelta(hours=1))

        min_temp = min(storage_temperature)
        plt.ylim(max(min_temp - 30, 0), max(105, max(storage_temperature)))

        plt.xlabel('Hour of Day')
        plt.ylabel('Temperature (°C)')
        plt.title('Storage Water Temperature Throughout the Day')
        plt.legend()
        plt.grid(True)

        # Show x-axis labels for every hour and add labels for the first and last hour manually if they are missing
        plt.xticks([date_times[0] + timedelta(hours=i) for i in range(25)], [f"{i}" for i in range(25)])

        plt.show()
    

    @staticmethod
    def plot_month(storage_temperature, date_times):
        """
            Plots the temeperature profile for month's simulation 
        """
        plt.figure(figsize=(12, 6))
        plt.plot(date_times, storage_temperature, color='blue', label='Storage Water Temperature')

        # Plotting a line parallel to the x-axis at y=60
        plt.axhline(y=60, color='red', linestyle='--', label='Load Temperature requirement = 60°C')
        plt.axhline(y=100, color='red', linestyle='--', label='Limiting Temerature Line = 100°C')

        # Setting the x-axis to show each day of the month
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d'))  # Show day of the month
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # One tick per day

        # Adding labels and title
        plt.xlabel('Day of Month')
        plt.ylabel('Temperature (°C)')
        plt.title('Storage Temperature Throughout the Month')
        min_temp = min(storage_temperature)
        plt.ylim(max(min_temp - 30, 0), max(105, max(storage_temperature)))

        # Enabling grid for better readability
        plt.grid(True)
        plt.xlim(date_times[0], date_times[-1])

        # Adding legend
        plt.legend()

        # Show plot
        plt.show()

    # TODO: CAN this be made nicer?
    @staticmethod
    def plot_year(storage_temperature, date_times):
        """
            Plots the temeperature profile simulation of the entire year
        """
        plt.figure(figsize=(12, 6))
        plt.plot(date_times, storage_temperature, color='blue', label='Storage Water Temperature')

        # Plotting a line parallel to the x-axis at y=60
        plt.axhline(y=60, color='red', linestyle='--', label='Load Temperature requirement = 60°C')
        plt.axhline(y=100, color='red', linestyle='--', label='Limiting Temerature Line = 100°C')
        # Setting the x-axis to show each day of the month
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        plt.xlabel('Month')
        plt.ylabel('Temperature (°C)')
        plt.title('Storage Temperature Throughout the Year')
        plt.xlim(date_times[0], date_times[-1])
        min_temp = min(storage_temperature)
        plt.ylim(max(min_temp - 30, 0), max(105, max(storage_temperature)))
        plt.grid(True)
        plt.legend()
        plt.show()
    
    @staticmethod
    def format_energy_values(energy_auxiliary, total_energy):
        # Define the energy unit thresholds
        thresholds = {
            'J': 1e3,      # Up to 1,000 J, use J
            'kJ': 1e6,     # Up to 1,000,000 J, use kJ
            'MJ': 1e9,     # Up to 1,000,000,000 J, use MJ
            'GJ': 1e12,    # Greater than 1,000,000,000,000 J, use GJ
        }
        max_value = max(total_energy, energy_auxiliary)
        unit = 'J'  # Default to J
        for key, value in thresholds.items():
            if max_value < value:
                unit = key
                break
        factor = 1e3 ** ('J kJ MJ GJ'.split().index(unit))
        total_energy_converted = total_energy / factor
        energy_auxiliary_converted = energy_auxiliary / factor
        total_energy_formatted = f"{total_energy_converted:.3f} {unit}"
        energy_auxiliary_formatted = f"{energy_auxiliary_converted:.3f} {unit}"
        return total_energy_formatted, energy_auxiliary_formatted


    def simulate_single_day(self, month=4, day=15):
        simulation_start_datetime = datetime(self.simulation_year, month, day)
        simulation_period = self._SECONDS_IN_A_DAY

        storage_temperature, total_energy_consumed, energy_met_by_auxiliary, date_times =  self._run_sumilation(simulation_start_datetime, simulation_period)
        total, aux  = self.format_energy_values(sum(energy_met_by_auxiliary), sum(total_energy_consumed))
        print("AUXILIARY ENERGY",  total)
        print("TOTAL_ENERGY", aux)
        print(f'THE SOLAR FRACTION IS: {(1 - sum(energy_met_by_auxiliary)/sum(total_energy_consumed)):.3f}')

        self.plot_solar_radiation(self.environment_conditions, month=month, day=day)
        self.plot_single_day(storage_temperature, date_times)
        self.plot_energy_bar_graph(total, aux)

    def simulate_month(self, month=1):
        simulation_start_datetime = datetime(self.simulation_year, month, 1)
        # Calculate the number of days in the month
        days_in_month = calendar.monthrange(self.simulation_year, month)[1]
        # Calculate the simulation_period in seconds
        simulation_period = days_in_month * self._SECONDS_IN_A_DAY
        print("Month Simulation _ start_date_time", simulation_start_datetime)
        print("Month Simulation _ days in month", days_in_month)
        print("Month Simulation _ simulation period", simulation_period)
        storage_temperature, total_energy_consumed, energy_met_by_auxiliary, date_times = self._run_sumilation(simulation_start_datetime, simulation_period)
        total, aux  = self.format_energy_values(sum(energy_met_by_auxiliary), sum(total_energy_consumed))
        print("AUXILIARY ENERGY",  total)
        print("TOTAL_ENERGY", aux)
        print(f'THE SOLAR FRACTION IS: {(1 - sum(energy_met_by_auxiliary)/sum(total_energy_consumed)):.3f}')
        self.plot_solar_radiation(self.environment_conditions, month=1)
        self.plot_month(storage_temperature, date_times)
        self.plot_energy_bar_graph(total, aux)


    def simulate_entire_year(self):
        def get_seconds_in_a_year(year):
            """Returns the number of seconds in a year"""
            if calendar.isleap(year):
                return 366 * 24 * 60 * 60
            else:
                return 365 * 24 * 60 * 60
        simulation_start_datetime = datetime(self.simulation_year, 1, 1)
        simulation_period = get_seconds_in_a_year(self.simulation_year)
        storage_temperature, total_energy_consumed, energy_met_by_auxiliary, date_times = self._run_sumilation(simulation_start_datetime, simulation_period, year=True)
        total, aux  = self.format_energy_values(sum(energy_met_by_auxiliary), sum(total_energy_consumed))
        print("AUXILIARY ENERGY",  total)
        print("TOTAL_ENERGY", aux)
        print(f'THE SOLAR FRACTION IS: {(1 - sum(energy_met_by_auxiliary)/sum(total_energy_consumed)):.3f}')
        self.plot_solar_radiation(self.environment_conditions)
        self.plot_year(storage_temperature, date_times)
        self.plot_energy_bar_graph(total, aux)


    def _run_sumilation(self,  start_datetime, period, year=False):

        simulation_params = self.simulation_params

        # Initialize steady state starting temperature
        storage_water_temerature = [self._STEADY_STATE_STARTING_TEMPERATURE]
        delta_t = self._SIMULATION_STEP if not year else 15 * 60
        date_times = []
        total_energy_consumed = []
        energy_met_by_auxiliary = []
        
        for timestep in range(0, period, delta_t):
            # Calculate the current simulation time
            current_date_time = start_datetime + timedelta(seconds=timestep)
            date_times.append(current_date_time)
            # Get current solar radiation and ambient temperature
            I_t, T_a = self.get_solar_radiation_and_temperature(current_date_time, simulation_params.environment_conditions)
            # print(I_t, T_a)
            # print("DELTA_T", delta_t)
            current_time = current_date_time.time()
            is_consumed = self.is_water_consumed(current_time, simulation_params.consumption_pattern)
            # print(is_water_consumed(current_time, water_consumption))
            # print("CRRTIME", current_time)
            if not is_consumed:
                storage_water_temerature.append(storage_water_temerature[-1])
                total_energy_consumed.append(0)
                energy_met_by_auxiliary.append(0)
            else:
                storage_water_temerature.append(self.get_next_temperature(simulation_params, I_t, storage_water_temerature[-1], simulation_params.T_l, T_a, delta_t))
                total_energy_consumed.append(self.calculate_total_energy_rate(simulation_params, T_a) * delta_t)
                aux = self.calculate_energy_rate_met_by_auxiliary(simulation_params, storage_water_temerature[-1], T_a) * delta_t
                energy_met_by_auxiliary.append(aux)

        return storage_water_temerature[1:], total_energy_consumed, energy_met_by_auxiliary, date_times