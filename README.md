# Simulating a Solar Water Heater

Program for simulating the heat transfer from a solar panel to a storage tank. The system components used are:
- **Environment** - The sun provides solar radiation
- **Solar Collector** - Transfers solar energy to storage tank by circulating water through collector tubes
- **Storage Tank** - Stores hot water, and supplies it to the load when there is a demand for hot water
- **Solar Pump** - Pumps water from storage tank to solar collector tubes
- **Make up water supply** - A make up water supply is assumed, which supplies water at ambient temerature to the stroage tank (As soon as water is used up, ensures tank is **always full**)
- **Auxiliary heater** - An auxiliary water heater is assumed, which provides the the differential energy requirements to supply hot water at the desired temerature. 

<img src="https://github.com/yashladd/solar_water_heater/blob/main/System_component.png" alt="System Components">

# Code Walkthrough

To run a simulation:

Create a `config.json` file (see instructions below)

Import the `SimulationEnvironment` class from `src/models.py`. It takes 3 parameters:

1. `collector_area`
2. `storage_tank_volume`
3. `path_to_your_config_file`

The `SimulationEnvironment` class exposes 3 main functions:

1. `simulate_single_day(starting_temerature, month=4, day=15)`
2. `simulate_month(starting_temerature, month=1)`
3. `simulate_entire_year(starting_temerature)`

## Outputs
The simulations are ran for the simulation period in intervals of timestep $t= 300$ seconds.

1. A graph showing the solar radiation during the simulation period
2. A graph showing the simulation of the Temerature of the tank for the simulation period.
3. A graph showing the energy consumption and Solar Fraction for the simulation period.

# How to Use: 

#### Install the dependencies
1. Clone the repo:
    ```sh
    git clone git@github.com:yashladd/solar_water_heater.git
    ```
2. Create a `python` virualenvironment [Optional]
3. pip install requirement:

    ```sh
    pip install -r requirements.txt
    ```

#### Run the exaplmes 
I'm using `Jupyter` a notebooks to showcase the simulations ran in varying weather conditions and water consumption patterns. Example simulations can be found in the `examples` directory.
1. [An Example showcasing the simulation of a city in India](https://github.com/yashladd/solar_water_heater/blob/main/examples/example_india.ipynb)
2. [An Example showcasing the simulation of Salt Lake City](https://github.com/yashladd/solar_water_heater/blob/main/examples/example_slc.ipynb)
3. [An Example showcasing demand of continuous hot water supply throughout the day in India](https://github.com/yashladd/solar_water_heater/blob/main/examples/example_water_continuous.ipynb)

#### Create your own configuration and run simulations

<details>
<summary><b>Show instructions</b></summary>

The required conditions for simulation can be defined here. For example, the latitude and longitude of the environment, the year to simulate, water consumption patterns, load temperature, and other properties can be defined here. Here is a detailed example of a config.json:

```json
{
    "simulation_year": 2006,
    "environment": {
        "latitude": 18.53,
        "longitude": 73.85,
        "timezone": "Asia/Kolkata"
    },
    "solar_collector": {
        "f_r_tao_alpha": 0.675,
        "f_r_u_l": 5.656,
        "tilt": 33
    },
    "load_profile": {
        "litres_per_day": 4500,
        "water_density": 998,
        "specific_heat_water": 4180,
        "desired_temperature": 60,
        "consumption_pattern": [
            {
                "start": "06:00",
                "end": "08:00"
            },
            {
                "start": "10:00",
                "end": "11:00"
            },
            {
                "start": "12:00",
                "end": "13:00"
            },
            {
                "start": "14:00",
                "end": "18:00"
            }
        ]
    },
    "storage_tank": {
        "type": "Cylindrical",
        "height_to_diameter_ratio": 1,
        "wall_material": "Mild steel",
        "storage_wall_thicknes": 0.006,
        "thermal_conductivity_wall": 50,
        "insulation_material": "Glass wool",
        "insulation_thickness": 0.2,
        "insulation_thermal_conductivity": 0.04
    }
}
```

Having created your desired `config.json` file, create a jupyter notebook and import the `SimulationEnvironment` class and `config.json` file.

```python
import os
import os.path as path
import sys

EXAMPLE_DIR = path.abspath("")
GIT_DIR = path.split(EXAMPLE_DIR)[0]
SRC_DIR = path.join(GIT_DIR, "src")
sys.path.append(SRC_DIR)

CONFIG_FILE = f'{EXAMPLE_DIR}/your_config.json'
from models import SimulationEnvironment

AREA = 50
VOLUME = 5
# Instantiate simulation class
simulation = SimulationEnvironment(AREA, VOLUME, CONFIG_FILE)

# Run your simulations
```

</details>




# Potential Use Case

For a given solar collector-storage system, parameters such as **collector area**, **storage volume** and **solar fraction** are crucial from the performance and optimization point of view. The program aims to **estimate** these design parameter in a given **environment** to meet the sepecied **demand** of hot water (desired water temperature and water consumption pattern).

# Thermodynamic Approach
Since this system cannot be designed for a two-phase condition, it requrires that the hot water temerature in the storage tank never exceeds $100 °C$, the boiling temerature of water (working fluid) to avoid steam formation. 
$T_{st} \leq T_{sat}$ ($100 °C$). 

## Terminology
- $\rho$  -  density of working fluid(water), $kg/m^3$
- $C_p$   -  specific heat of working fluid, $J/kg°C$
- $V_{st}$ - Volume of the storage tank, $m^3$
- $T_{st}$ - storage temperature $°C$
- $T_a$ - ambient temperature, $°C$
- $T_L$ - desired load (hot water) temperature, $°C$
- $U_{st}$ - storage heat loss coefficient, $W/m^2°C$
- $A_{st}$ surface area of the storage tank, $m^2$
- $F_R$ - collector heat removal factor
- $U_L$ collector overall heat loss coefficient, $W/m^2°C$ 
- $\tau\alpha$ - average transmittance absorptance product
- $I_t$ - solar radiation intensity on tilted surface, $W/m^2$
- $A_c$ - collector area, $m^2$
- $\dot{m_l}$ - desired load mass flow rate, $kg/s$
- $q_s$ - solar useful heat gain rate, $W$
- $q_{Ls}$ - load rate met by solar energy, $W$
- $q_{stl}$ - rate of storage loss, $W$
- $q_{aux}$ rate of auxiliary energy required, $W$
- $Q_L$ desired hot water load over a specified time horizon, $J$
- $Q_{Ls}$ load met by solar energy over a specified time horizon, $J$
- $Q_{aux}$ - auxiliary energy required over a specified time
horizon, $J$
- $t$ time step in the simulation analysis, $s$

## Simulating the Storage tank temerature $T_{st}$
**Storage tank temperature** ($T_{st}$) is an important parameter
which influences the system size and performance. In my simulation, I focus on modeling the temperature dynamics of the storage tank across different time frames—ranging from a single day to a month, or spanning an entire year. The total energy required by the system and energy demands met by auxiliary heater are tracked during the simulation.


Energy balance of a well mixed storage tank over a time horizon can be
expressed as

**Equation (1)**

$$ (\rho C_p V_{st}) \cdot \frac{d T_{st}}{dt} = q_s - q_{Ls} - q_{stl} $$

Here $q_s$, the solar useful heat gain rate, is calculated ([Duffie and Huffman](https://www.sku.ac.ir/Datafiles/BookLibrary/45/John%20A.%20Duffie,%20William%20A.%20Beckman(auth.)-Solar%20Engineering%20of%20Thermal%20Processes,%20Fourth%20Edition%20(2013).pdf)) as 

$$q_s = A_c \left[ I_t F_{R} (\tau\alpha) − F_{R} U_{L} (T_{st} - T_a) \right]^+ $$
where + indicates that only the positive values of $q_s$ will be considered in the analysis. This implies that hot water from
the collector enters the tank only when solar useful heat
gain becomes positive.

During demand, hot water is supplied at $T_L$, the desired load (hot water) temperature. In a time step $t$, when the current ambient temerature is $T_a$ the rate of heat supplied by solar energy $q_{Ls}$ can be written as:

**Equation (2)**

If, $T_{st} \geq T_L$

$$q_{Ls} = \dot{m_l} C_p (T_L - T_a)$$

If, $T_a \leq T_{st} < T_L $, the rate partial energy is supplied by the solar panel which is:

$$q_{Ls} = \dot{m_l} C_p (T_{st} - T_a)$$

Otherwise, 

$$q_{Ls} = 0$$

The rate of storage loss ($q_{stl}$) is estimated to be:

**Equation (3)**

$$q_{stl} = U_{st} \cdot A_{st} \cdot (T_{st} - T_a)$$ 


During **demand** for hot water, four different cases arise:
- $T_{st} \geq T_L$ and $q_s > 0$
- $T_{st} \geq T_L$ and $q_s < 0$
- $T_{st} < T_L$ and $q_s > 0$
- $T_{st} < T_L$ and $q_s < 0$

When there is **no** demand, $q_{Ls} = 0$ naturally, and the temerature $T_{st}$ is modeled only based on $q_s$ and $q_{stl}$ 

Using these equations, at the current timestep $t$, depending on the **current state** of the system, Equation $(1)$ can be integrated over the time step $t$ and by substituting the appropriate values of $q_s$, $q_{Ls}$ and $q_{stl}$ (from Equation(2) & Euqation(3)). The final temerature of the storage tank $T_{stf}$ is calculated when the initial temeprature at the start of the time step $t$ is $T_{sti}$. 

For instance when $T_{st} \geq T_L$ and $q_s > 0$, the analalytical equation derived by integrating Equation (1) and substituting appropriate values, is given by:


$$\frac{A_c I_t F_{R}(\tau\alpha) - A_c F_{R} U_{L}(T_{stf} - T_a) - \dot{m_l} C_p (T_{l} - T_a) - U_{st} A_{st}(T_{stf} - T_a)}{A_c I_t F_{R}(\tau\alpha) - A_c F_{R} U_{L}(T_{sti} - T_a) - \dot{m_l} C_p (T_{l} - T_a) - U_{st} A_{st}(T_{sti} - T_a)} = \exp \left(-\frac{(A_c F_{R} U_{L} + U_{st} A_{st}) \cdot t}{\rho C_p V_{st}}\right)$$

Equations are similartly derived for **all** the cases mentioned above and their analytical solutions are used to model the storage tank temperature **$T_{st}$**

<details>
<summary><b>Show all equations</b></summary>

Using the energy balance of a well mixed storage tank over a time horizon $t$:


$$(\rho C_p V_{st}) \cdot \frac{d T_{st}}{dt} = q_s - q_{Ls} - q_{stl}$$

The temperature profile $T_{st}$ is modeled under various scenarios, resulting in 6 distinct cases. The analytical solutions for these cases are:

$T_{stf}$ is the final temerature at the end of the time step $t$ and $T_{sti}$ is the inital temerature.

When there is **demand** for hot water:

1. $T_{st} \geq T_L$ and $q_s > 0$ :
        $$\frac{A_c I_t F_{R}(\tau\alpha) - A_c F_{R} U_{L}(T_{stf} - T_a) - \dot{m_l} C_p (T_{l} - T_a) - U_{st} A_{st}(T_{stf} - T_a)}{A_c I_t F_{R}(\tau\alpha) - A_c F_{R} U_{L}(T_{sti} - T_a) - \dot{m_l} C_p (T_{l} - T_a) - U_{st} A_{st}(T_{sti} - T_a)} = \exp \left(-\frac{(A_c F_{R} U_{L} + U_{st} A_{st}) \cdot t}{\rho C_p V_{st}}\right)$$

2. $T_{st} \geq T_L$ and $q_s < 0$:
        $$\frac{ - \dot{m_l} C_p (T_{l} - T_a) - U_{st} A_{st}(T_{stf} - T_a)}{ - \dot{m_l} C_p (T_{l} - T_a) - U_{st} A_{st}(T_{sti} - T_a)} = \exp \left(-\frac{(U_{st} A_{st}) \cdot t}{\rho C_p V_{st}}\right)$$

3. $T_{st} < T_L$ and $q_s > 0$:
        $$\frac{A_c I_t F_{R}(\tau\alpha) - A_c F_{R} U_{L}(T_{stf} - T_a) - \dot{m_l} C_p (T_{stf} - T_a) - U_{st} A_{st}(T_{stf} - T_a)}{A_c I_t F_{R}(\tau\alpha) - A_c F_{R} U_{L}(T_{sti} - T_a) - \dot{m_l} C_p (T_{stf} - T_a) - U_{st} A_{st}(T_{stf} - T_a)} = \exp \left(-\frac{(A_c F_{R} U_{L} + U_{st} A_{st} + \dot{m_l} C_p) \cdot t}{\rho C_p V_{st}}\right)$$

4. $T_{st} < T_L$ and $q_s < 0$:
        $$\frac{ - \dot{m_l} C_p (T_{stf} - T_a) - U_{st} A_{st}(T_{stf} - T_a)}{ - \dot{m_l} C_p (T_{sti} - T_a) - U_{st} A_{st}(T_{sti} - T_a)} = \exp \left(-\frac{(U_{st} A_{st} + \dot{m_l} C_p) \cdot t}{\rho C_p V_{st}}\right)$$


When there is **no demand** for hot water

5. $q_s < 0$
        $$\frac{T_{stf} - T_a}{T_{sti} - T_a} = \exp \left( -\frac{(U_{st} A_{st}) \cdot t}{\rho C_p V_{st}} \right)$$

6. $q_s > 0$
        $$\frac{A_c I_t F_{R}(\tau\alpha) - A_c F_{R} U_{L}(T_{stf} - T_a) - U_{st} A_{st}(T_{stf} - T_a)}{A_c I_t F_{R}(\tau\alpha) - A_c F_{R} U_{L}(T_{sti} - T_a) - U_{st} A_{st}(T_{sti} - T_a)} = \exp \left(-\frac{(A_c F_{R} U_{L} + U_{st} A_{st}) \cdot t}{\rho C_p V_{st}}\right)$$

</details>


## Calculating Solar Fraction
Solar fraction, is defined as the percentage of energy provided by solar divided by the total energy required.

The total energy required by the system over the course of the entire simulation will be:

$$ Q_L = \sum \dot{m_l} C_p (T_{l} - T_a) \cdot t $$

where $t$ is the time step used in the simulation. 


Based on equations for $q_{Ls}$ (Equation (2)), the rate of energy demands met by the auxiliary can be calculated in that time step. 

$$ q_{aux} = q_{L} - q_{Ls} $$

The total energy met by the auxiliary over the over the course of the entire simulation will be:

$$Q_{aux} = \sum \left[ (q_L - q_{Ls})  \cdot t \right]^+ $$

Where the + sign indicates that only the positive values of the equation are taken, meaning the auxiliary is only used when the solar energy is not fully able to meet the load demand ( $T_{st} < T_L$ ). 

Therefore the solar fraction ($F$) can be calculated as:

$$
    F = \frac{Q_{Ls}}{Q_L} = 1 - \frac{Q_{aux}}{Q_L}
$$

In this way the solar fraction $F$ can be calculated for the entire simulation by summing up the energies over the all the time steps $t$.


## Solar radiation and Weather Conditions ($I_t$ and $T_a$)
In my simulation, **historical hourly** weather data is utilized to derive values for solar radiation on a tilted surface ($I_t$) and ambient temperature ($T_a$). These metrics are obtained using the **[pvlib](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.iotools.get_pvgis_hourly.html)** Python package, which offers flexibility in terms of adjusting for various geographical locations and different tilts of solar collector. It provides historical data forom **2005-2015**

## Storage loss coefficient ($U_{st}$)

I have estimated $U_{st}$, the storage heat loss coefficient, based on thermal resistance properties of the tank and the thickness of its wall and the insulation used which are predefined in the **[config](https://github.com/yashladd/solar_water_heater/blob/main/examples/config_india.json#L41)**

## Assumptions 
1. The storage tank is always assumed to be well mixed and always full.
2. Heat loss within the pipes of the pump that circulate water in the system is neglected.
3. The ambient temperature $T_a$ and the solar radiation $I_t$ are assumed to be constant within a window of $1 hr$.
4. Make up water supply is assumed to be at ambient temerature. 

## Constraints
1. The temerature of the storage tank must never exceed the boiling temerature of water ($100 °C$).
2. Currently can only run simulations from year 2005-2015.

## Future work
1. Calculating cost for various system components.
2. For a desired solar fraction $F$ in a given environment, determine the optimal colelctor area $A_c$ and volume of the storage tank $V_{st}$. This will involve optimizing a multi-objective function and will aim to reduce total cost for designing the system.

## References
1. [Govind   N.   Kulkarni,   Shireesh   B.   Kedare,   Santanu   Bandyopadhyay,  “Determination  of  Design  Space  and  Optimization  of  Solar  Water  Heating  Systems”,  Solar  Energy, Vol. 81, pp. 958-968, 2007.](https://www.sciencedirect.com/science/article/pii/S0038092X06003112)
2. [Duffie, J.A., Beckman, W.A., 1991. Solar Engineering of Thermal
Processes, second ed. Wiley, New York, pp. 686–732.](https://www.sku.ac.ir/Datafiles/BookLibrary/45/John%20A.%20Duffie,%20William%20A.%20Beckman(auth.)-Solar%20Engineering%20of%20Thermal%20Processes,%20Fourth%20Edition%20(2013).pdf)
3. [Solar heating design, by the f-chart method, Beckman, W. A., Klein, S. A, Duffie, J. A. NSF, ERDA, and University of Wisconsin. New York, Wiley-Interscience, 1977. 214 pp.](https://search.worldcat.org/title/solar-heating-design-by-the-f-chart-method/oclc/3168307)