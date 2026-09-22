# Data and target

The solar target is normalized AC PV power (kW/kWp), constructed as a simplified physics-based reference from NASA POWER inputs. It is not measured plant production.

Target construction uses solar position, isotropic plane-of-array irradiance, a simplified NOCT cell-temperature relation, an irradiance-proportional DC response with temperature correction, constant inverter efficiency, and DC/AC clipping. Adopted constants include temperature coefficient -0.004 /°C, inverter efficiency 0.96, DC/AC ratio 1.20, albedo 0.20, and NOCT 45 °C.

Plane-of-array irradiance and estimated cell temperature are target-construction intermediates and are not part of the retained 53-predictor LSTM forecasting input.
