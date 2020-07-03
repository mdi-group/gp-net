* Stareted from `../../band_gap_data.pkl`
* Shuffled randomly
* Data for AL = shuffled[:30000]
    * After removing zero bandgap = 22695
* Data for final tests = shuffled[30000:]
* AL cycle:
    * Start with 1 points in the pool
    * Train MegNet for 100 iterations
    * Split points 80:20 for training GP
    * Train GP for 100 iterations
    * After each cycle add 200 extra points 

* Command: `python3 gp-net.py -meg -data band_gap_data.pkl -frac 0.045 0.2 -epochs 100 -samp 200 10 -amp 10 -length 10 -maxiters 100`
