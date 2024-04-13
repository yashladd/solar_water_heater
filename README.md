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
## Terminology
- $\rho$  -  density of working fluid(water), $kg/m^3$
- $C_p$   -  specific heat of working fluid, $J/kg°C$
- $V_{st}$ - Volume of the storage tank, $m^3$
- $T_{st}$ - storage temperature $°C$
- $T_a$ - ambient temperature, $°C$
- $T_L$ - desired load (hot water) temperature, $°C$
- $U_{st}$ - storage heat loss coefficient, $W/m^2°C$
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
- $t$ time step in the simulation analysis, $s$

<!-- - $\dot{m_r}$ - makeup water mass flow rate, $kg/s$ -->
<!-- - $\dot{m_s}$ - mass flow rate from storage, $kg/s$ -->


<!-- The temperature of the storage tank ($T_{st}$) is a critical factor affecting both the sizing of the system and its operational efficiency. In my simulation, I focus on modeling the temperature dynamics of the storage tank across different timeframes—ranging from a single day to a month, or even spanning an entire year. Key parameters such as the total area of the solar collector (\(A_c\)), the volume of the storage tank (\(V_{st}\)), and the solar fraction (\(F\)) are central to optimizing the system for both performance and cost-effectiveness. -->
<!-- $$(\rho C_p V_{st}) \cdot \frac{d T_{st}}{dt}  = q_s - q_{Ls} - q_{stl}$$ -->
<!-- $TODO$ fix the link -->
## Simulating the Storage tank temerature $T_{st}$
**Storage tank temperature** ($T_{st}$) is an important parameter
which influences the system size and performance. In my simulation, I focus on modeling the temperature dynamics of the storage tank across different timeframes—ranging from a single day to a month, or even spanning an entire year. The total collector area $A_c$ and storage tnak volume $V_{st}$ and the solar fraction $F$,  are important from the performance and cost optimization point of view. 


Energy balance of a well mixed storage tank over a time horizon can be
expressed as

**Equation (1)**

$$ (\rho C_p V_{st}) \cdot \frac{d T_{st}}{dt} = q_s - q_{Ls} - q_{stl} $$

Here $q_s$, the solar useful heat gain rate, is calculated ([Duffie and Huffman](https://google.com)) as 

$$ q_s = A_c \cdot [I_tF_{R}(\tau\alpha) − F_{R}U_{L}(T_{st} - T_a)]^+$$
where + indicates that only the positive values of $q_s$ will be considered in the analysis. This implies that hot water from
the collector enters the tank only when solar useful heat
gain becomes positive.

During demand, hot water is supplied at $T_L$, the desired load (hot water) temperature. In a time step $t$, when the current ambient temerature is $T_a$ the rate of heat supplied by solar energy $q_{Ls}$ can be written as:

if $T_{st} \geq T_L$

$$
    q_{Ls} = \dot{m_l} C_p (T_L - T_a)
$$

If $T_a \leq T_{st} < T_L $, the rate partial energy is supplied by the solar panel which is:

$$q_{Ls} = \dot{m_l} C_p (T_{st} - T_a)$$

Otherwise, 

$$q_{Ls} = 0$$

The rate of storage loss ($q_{stl}$) is estimated to be

$$q_{stl} = U_{st} \cdot A_{st} \cdot (T_{st} - T_a)$$ 


During **demand** for hot water, four different cases arise:
- $T_{st} \geq T_L$ and $q_s > 0$
- $T_{st} \geq T_L$ and $q_s < 0$
- $T_{st} < T_L$ and $q_s > 0$
- $T_{st} < T_L$ and $q_s < 0$

When there is **no** demand, $q_{Ls} = 0$ naturally, and the temerature $T_{st}$ is models only based on $q_s$ and $q_{stl}$ 

Using these equations, at the current timestep $t$, depending on the state of the system, Equation $(1)$ can be integrated over the time step $t$ and substituting the appropriate values of $q_s$, $q_{Ls}$ and $q_{stl}$. The final temerature of the storage tank $T_{stf}$ is calculated when the initial temeprature at the start of the time step $t$ is $T_{sti}$. 

For instance when $T_{st} \geq T_L$ and $q_s > 0$, the analalytical equation derived by integrating Equation (1) and substituting appropriate values, is given by:


$$\frac{A_c I_t F_{R}(\tau\alpha) - A_c F_{R} U_{L}(T_{stf} - T_a) - \dot{m}_L C_p (T_{l} - T_a) - U_{st} A_{st}(T_{stf} - T_a)}{A_c I_t F_{R}(\tau\alpha) - A_c F_{R} U_{L}(T_{sti} - T_a) - \dot{m}_L C_p (T_{l} - T_a) - U_{st} A_{st}(T_{stf} - T_a)} = \exp (-\frac{(A_c F_{R} U_{L} + U_{st} A_{st}) \cdot t}{\rho C_p V_{st}})$$

Equations are similartly derived for all the scenarios mentioned above and their analytical solutions are used to model the storage tank temperature $T_{st}$

## Calculating Solar Fraction




In my simulation, historical hourly weather data is utilized to derive values for solar irradiance ($I_t$) and ambient temperature ($T_a$). These metrics are obtained using the **[pvlib](https://pvlib-python.readthedocs.io/en/stable/)** Python package, which offers flexibility in terms of adjusting for various geographical locations and different tilts of solar collector.



I have estimated $U_{st}$, the storage heat loss coefficient, based on thermal resistance properties of the tank and the thickness of its wall and the insulation used these can be predefined in the $TODO put link here$ [config](https://google.com). 


The total energy rate for meeting this demand is, at a particulart time step is:




If the temerature of the storage tank $T_{st} \geq T_L$, the demand is met entirely by the solar energy $q_{Ls}$. Otherwise, the ramaining energy is supplied by the auxiliary heater. 

