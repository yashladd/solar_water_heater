# Simulating a Solar Water Heater

Program for simulating the heat transfer from a solar panel to a storage tank. The system components used are:
- Solar Collector - transfers solar energy to storage tank by circulating water through collector tubes
- Storage Tank - stores hot water
- Solar Pump - pumps water from storage tank to solar collector tubes
- Make up water supply - Supplies cold water to the stroage tank (As soon as water is used up, ensures tank is always full)
- Auxiliary heater - This is used when the solar collector is unable to meet the temperature demand of the Load
- Environment - The sun provides solar radiation


<img src="https://github.com/yashladd/solar_water_heater/blob/main/system_components.png" alt="System Components">

# How to Use: 
TODO: Yash 


# Potential Use Case

For a given solar collector-storage system, parameters such as **collector area**, **storage volume** and **solar fraction** are crucial from the performance and optimization point of view. The program aims to determine these design parameter in a given **environment** to meet the sepecied **demand** (desired water temperature and water consumption pattern).

# Thermodynamic Approach
## Nomenclature
- $\rho$  -  density of working fluid(water), $kg/m^3$
- $C_p$   -  specific heat of working fluid, $J/kg°C$
- $V_{st}$ - Volume of the storage tank, $m^3$
- $T_{st}$ - storage temperature $°C$
- $T_a$ - ambient temperature, $°C$
- $T_L$ - desired load (hot water) temperature, $°C$
- $U_{st}$  -storage heat loss coefficient, $W/m^2°C$
- $A_{st}$ surface area of the storage tank, $m^2$
- $q_s$ - solar useful heat gain rate, $W$
- $q_{Ls}$ - load met by solar energy, $W$
- $q_{stl}$ - rate of storage loss, $W$
- $F_r$ - collector heat removal factor
- $\tau\alpha$ - average transmittance absorptance product
- $I_t$ - solar radiation intensity on tilted surface, $W/m^2$
- $A_c$ - collector area, $m^2$
- $U_l$ - collector overall heat loss coefficient, $W/m^2°C$
- $\dot{m_l}$ - desired load mass flow rate, $kg/s$
- $\dot{m_r}$ - makeup water mass flow rate, $kg/s$
- $\dot{m_s}$ - mass flow rate from storage, $kg/s$
- $t$ time step in the simulation analysis, $s$

**Storage tank temperature** ($T_{st}$) is an important parameter
which influences the system size and performance. My approach is to model the storage tank temerature over a period of a single day or a month or the entire year. The collector area $A_c$ and storage tnak volume $V_{st}$

Energy balance of a well mixed storage tank can be
expressed as

$$\frac{\rho C_p V_{st}}{dt} = q_s - q_{Ls} - q_{stl}
$$
