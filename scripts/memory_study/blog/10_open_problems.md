# Open problems

Two directions we have not explored, both of which could move the numbers:

- **The ideal roofline of DistMuon.** The optimizer step is almost all
  Newton-Schulz iterations ([[compute_roofline.2.3]]), and each iteration
  multiplies a matrix by its own transpose. That product is symmetric, so a
  dedicated kernel needs to compute only half of it. We take DistMuon's
  kernels as they run today; a roofline built from the work Newton-Schulz
  really needs would show how much a faster optimizer could save.
- **Other pipeline schedules.** We modelled plain and interleaved 1F1B. Splitting
  each backward into its input-gradient and weight-gradient halves lets a
  schedule fill the bubble with weight-gradient work: zero-bubble schedules,
  and V-shaped ones (each rank holding a chunk near each end of the pipeline)
  that balance memory across ranks. Each offers a different trade between
  bubble and memory, and with attention residuals, a different pattern of what
  each hop carries ([[step]]).

## Notes
