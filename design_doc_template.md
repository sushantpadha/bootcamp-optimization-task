# Design Document — Monster Spawning Grid
**Name:** ___________________  
**Date:** ___________________  
**Final median time (10 runs, public_1):** ___________________ ms  
**Reference median time (10 runs, public_1):** ___________________ ms  
**Speedup:** ___________________×

---

## 1. Cell Representation

*What internal format do you use to store the grid during simulation?  Explain the trade-offs you considered (bytes, bit-packing, bitplane decomposition, other) and why you chose what you did.*

---

## 2. Parallelisation Strategy

*How do you distribute work across the 8 cores?  Thread pool, `std::execution`, raw `std::thread`?  How do you partition the grid?  How do you handle the toroidal halo at tile boundaries?*

---

## 3. SIMD Strategy

*Do you use SIMD?  If yes: NEON or SVE2, and why?  What is the key operation you vectorise?  If no: why not, and what did that leave on the table?*

---

## 4. Memory Layout and Tiling

*What is your tile size?  How did you choose it?  What cache level were you targeting?  Did you use temporal blocking (multiple generations per tile)?*

---

## 5. What Didn't Work

*At least two things you tried that made things worse or no better.  Include enough detail that a reader could reproduce the experiment.*

---

## 6. What You Would Do With Another Week

*One concrete optimisation you believe would yield a measurable speedup, and why.*

---

## 7. Benchmark Methodology

*How did you measure?  How many runs?  What did you control for (ASLR, CPU frequency, page-cache warmth)?  What was your run-to-run variance (CV)?  How did you isolate the effect of individual changes?*

---

*Maximum 3 pages total. Tables and figures count toward the limit.*
