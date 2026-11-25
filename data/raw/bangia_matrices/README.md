# Bangia et al. transition matrices

Source: Bangia, Diebold, Kronimus, Schagen and Schuermann, Ratings Migration and the Business Cycle, With Application to Credit Portfolio Stress Testing. Public draft, Tables 4.2 (printed p. 18) and 4.3 (printed p. 25).
PDF: https://www.planchet.net/EXT/ISFA/1226.nsf/c3f387ba64b12435c1256a16003fcc4b/a7bff8b3b05264f5c12575eb0019cc9d/%24FILE/bangia%20et%20al%20-%20transition%20matrix%20cycles%20eco.pdf
Published article: https://doi.org/10.1016/S0378-4266(01)00229-1

Each matrix has seven initial ratings (rows) and eight terminal ratings (columns). The published tables have no initial D row; none is added. Numeric entries are probabilities (published percentage divided by 100), with four decimal places. Dashes in observed rows are encoded as zero; all-dash rows with no issuer periods (up/CCC and down/AAA) are left blank and should be read as missing, not zero. Published rounding is preserved; no row normalization was performed. The issuer-period counts from Table 4.3 are supplied separately; dashes in counts for unobserved rows are encoded as 0. Significance highlighting is not encoded.

For equal-weight calibration across all momentum conditions, use common observed initial ratings AA, A, BBB, BB, B. Add an absorbing D row only when building a simulation transition matrix. These are historical data, not modern S&P baseline values.
