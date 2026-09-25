# Registered-claim map

`data/expected/reported_claims.csv` is the machine-readable expected-value and
tolerance registry. The manuscript and its artwork are intentionally excluded
from this repository; the locators below connect each selected claim family to
the submitted article. The registry emphasizes conclusions and inferential
boundaries and does not enumerate every number printed in the manuscript.

| Claim identifiers | Manuscript location |
|---|---|
| `source_gages`, `accepted_basins` | Section 2.1; Figure 1 |
| `estimable_basins`, `primary_transitions` | Section 3.1; Figure 1; Table 1 |
| `m_median`, `m_q25`, `m_q75`, `m_median_r2`, `m_random_effects_center`, `m_prediction_low`, `m_prediction_high`, `m_i2_percent` | Section 3.2; Table 1; Figures 4–5 |
| `tail_fraction_median`, `tail_multiple_median`, `excess_kurtosis_median` | Section 3.3; Figure 6 |
| `positive_tail_multiple_median`, `negative_tail_multiple_median` | Section 3.3; Figures 9b and 12b–c |
| `theory_pair_basins`, `decline_m_median`, `two_b_median`, `decline_minus_two_b_median`, `gamma_median`, `gamma_ci_below_zero_fraction`, `negative_within_share_median_paired` | Section 3.4; Figure 7 |
| `state_gaussian_crps_gain_*`, `empirical_shape_crps_gain_*` | Section 3.5; Figures 3d and 10 |
| `state_spatial_supported_*`, `state_water_year_supported_*`, `empirical_spatial_supported_*`, `empirical_water_year_supported_*` | Section 3.5; Figures 3d and 10; Supporting Figure S5 |
| `lambda_pacific_northwest_*`, `lambda_texas_gulf_*`, `gamma_pacific_northwest_*`, `gamma_texas_gulf_*`, `negative_share_south_atlantic_gulf_*`, `negative_share_pacific_northwest_*` | Section 3.5; Figure 9c–e |
| `regional_positive_tail_*`, `regional_negative_tail_*`, `regional_all_tail_medians_above_one` | Section 3.5; Figure 9b |
| `prediction_california_*`, `prediction_upper_colorado_*`, `prediction_south_atlantic_gulf_*`, `prediction_ohio_*` | Section 3.5; Figure 9f |

Wildcard suffixes identify every lead or statistic sharing the displayed
prefix; the verifier writes the expanded, row-by-row audit to
`outputs/verification/reported_claim_verification.csv`.
