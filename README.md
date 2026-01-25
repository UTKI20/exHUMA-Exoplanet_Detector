# AI_Exoplanet_Detector
The Problem NASA missions like Kepler generate millions of "light curves" (stellar brightness data). Identifying planets manually is a massive human bottleneck, as the signals are extremely rare (less than 1%) and often buried under heavy "stellar noise."

The Solution We built an automated discovery pipeline using 1D Convolutional Neural Networks (CNN).

Preprocessing: Uses Gaussian filters to strip away noise and highlight planetary "transits."

Class Balancing: Employs SMOTE to handle the extreme rarity of planet samples.

Deep Learning: The CNN automatically recognizes the signature "U-shaped" brightness dips of orbiting planets.

Impact This system replaces months of manual vetting with seconds of computation, enabling faster, more scalable discovery of Earth-like worlds in massive deep-space datasets.
