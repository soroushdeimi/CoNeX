# Numenta component examples

Runnable examples for the Thousand Brains Theory and predictive coding
behaviors in `conex.behaviors.*.numenta`.

Both scripts are CPU-only and finish in a few seconds.

## Running

The scripts import `conex` as an installed package. From a source checkout,
either install it in editable mode:

```bash
pip install -e .
python Example/numenta/htm_sequence_learning.py
```

or point `PYTHONPATH` at the repository root:

```bash
PYTHONPATH=. python Example/numenta/htm_sequence_learning.py
```

## `htm_sequence_learning.py`

The full HTM pipeline: symbols are encoded as sparse distributed
representations, the Spatial Pooler turns them into a stable set of active
columns, and Temporal Memory learns which element follows which.

The anomaly score starts at 1.0 (nothing is predicted yet) and falls to zero
once the sequence `A B C D` is learned:

```
epoch  mean anomaly  bursting columns
    0         1.000              10.0
    4         0.275               2.8
    7         0.000               0.0
   11         0.000               0.0
```

Then an unexpected symbol is injected. Note that the surprise costs *two*
steps, not one — breaking the context leaves the following element unpredicted
as well:

```
symbol  anomaly  bursting columns
     A    0.000                 0
     B    0.000                 0
     X    1.000                10
     D    1.000                10
```

## `predictive_coding.py`

One population wired as a predictive coding level:

```
TopDownPrediction -> ErrorUnit -> PrecisionWeighting
    -> FreeEnergyMinimization -> PredictiveCodingLearning
```

The higher level carries an unrelated random pattern, so the top-down weights
have to learn the mapping onto the sensory input. Error and free energy fall
together:

```
 step   mean |error|   mean precision   free energy
    0       0.33209           1.0000        2.6553
    2       0.06951           1.0090       -0.0257
   16       0.00000           1.0762       -1.1754
   39       0.00000           1.1784       -2.6270
```

Perturbing the sensory input makes the error jump again, which is the surprise
signal a full hierarchy would propagate upwards.

## A note on the API

These behaviors are driven by explicit `forward()` and `compute()` calls and
read state the caller sets (`activity`, `sp_input`, `tm_input_columns`) rather
than reading `neurons.spikes` and writing `neurons.I`. They sit alongside the
spiking pipeline rather than inside it, which is why both scripts step them by
hand instead of calling `net.simulate_iterations()`.
