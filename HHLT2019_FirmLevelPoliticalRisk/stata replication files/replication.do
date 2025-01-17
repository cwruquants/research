* =====================================================================
*
* Date	: June 26, 2019
* Paper	: Hassan, Hollander, Van Lent, and Tahoun 2019
*		  "Firm-Level Political Risk: Measurement and Effects"
*
* This do file reproduces the tables, with table numbering and titles 
* corresponding to those in the published version of the paper
*
* All output is sent to Stata's result window, with the exception of
* Table 8, on variance decomposition, which creates a tex file
*
* As for Table 9, bootstrapping of standard errors is currently
* commented out, as it takes a while to run
*
* =====================================================================

* Define username and drive
local username mschwede
local drive D
global folder "`drive':/Users/`username'/Dropbox/Policy uncertainty/New/replication_files/"
cd "${folder}"

* Datasets
global firmquarter_data "FullReplication_FirmQuarter.dta"
global firmyear_data "FullReplication_FirmYear.dta" 
global firmqtopicquarter_data "FullReplication_FirmTopicQuarter.dta"
global execcomp_data "FullReplication_ExecComp.dta"

/*
Please note: The following variables are set to missing and renamed to
MISSING[variablename]in the "RestrictedReplication_xxx.dta" files 
because they are proprietary:

 - FirmQuarter: sic sic2 sue d2at lat impvol_w_std volatility_w_std
	deltasales_w_100 pct_capex_guidance1_w_100 av_retW_100
 - FirmYear: sic sic2 sue d2at lat hiring_investm_w_100 av_retW_100
 - FirmTopicQuarter: sic sic2 lat
 - ExecComp: exec_fullname execid
 
The variables can be calculated based on data from the following data
providers (please refer to Appendix A2 for more details):
 - Compustat for sic sic2 d2at lat d2at deltasales_w_100
	sue hiring_investm_w_100
 - IBES for pct_capex_guidance1_w_100
 - CRSP for volatility_w_std av_retW_100
 - OptionMetrics for impvol_w_std
 - Compustat ExecuComp for exec_fullname execid
*/

* Define Table 8 tex file
global table8_vardecomp "table8.tex"

* =====================================================================
* 
* Reproduce Table 4: Validation: Implied and realized volatility
*
* =====================================================================

est clear
use ${firmquarter_data}, clear

* a) Run regressions for both panels

foreach outcome of varlist volatility_w_std impvol_w_std {

	*1) No fixed effects
	qui reg `outcome' PRisk_std lat, vce(cluster firm_id)
	est sto `outcome'_a
	
	*2) No fixed effects + time series of average PRisk
	qui reg `outcome' PRisk_std aPRisk_std lat, vce(cluster firm_id)
	est sto `outcome'_b
	
	*3) Time fixed effects
	qui reghdfe `outcome' PRisk_std lat, vce(cluster firm_id) ///
		absorb(i.cdateQ) keepsingletons
	est sto `outcome'_c
	
	*4) Time and sector fixed effects
	qui reghdfe `outcome' PRisk_std lat, vce(cluster firm_id) ///
		absorb(i.cdateQ i.sic2) keepsingletons
	est sto `outcome'_d
	
	*5) Time and firm fixed effects
	qui reghdfe `outcome' PRisk_std lat, vce(cluster firm_id) ///
		absorb(i.cdateQ i.firm_id) keepsingletons
	est sto `outcome'_e
	
	*6) Time , firm, and CEO fixed effects
	preserve
		* Add CEO data
		gen cdateY = yofd(dofq(cdateQ))
		joinby cdateY gvkey using "${execcomp_data}"
		gen quarter = quarter(dofq(cdateQ))
		*Regression
		qui reghdfe `outcome' PRisk_std lat if quarter == 1, vce(cluster firm_id) ///
			absorb(i.cdateQ i.ceo_id i.firm_id) keepsingletons
		est sto `outcome'_f
	restore
}

* b) Print output, panel by panel

esttab impvol_w_std_*, star(* 0.1 ** 0.05 *** 0.01) se
esttab volatility_w_std_*, star(* 0.1 ** 0.05 *** 0.01) se

* =====================================================================
* 
* Reproduce Table 5: Managing political risk
*
* =====================================================================

est clear
use ${firmquarter_data}, clear

* a) Run regressions for panel A

local i = 0
foreach outcome in capital_investm_w_100 pct_capex_guidance1_w_100 hiring_investm_w_100 ///
	deltasales_w_100 {
	
	local i = `i' + 1
	
	* Employment uses annual data
	if "`outcome'" == "hiring_investm_w_100" {
		preserve
			use ${firmyear_data}, clear
			qui reghdfe `outcome' PRisk_std lat, vce(cluster firm_id) ///
				absorb(i.sic2 i.cdateY) keepsingletons
			est sto panelA_`i'
		restore
	}
	
	* All other variables use quarterly data
	else {
		qui reghdfe `outcome' PRisk_std lat, vce(cluster firm_id) ///
			absorb(i.sic2 i.cdateQ) keepsingletons
		est sto panelA_`i'
	}
}

* b) Run regressions for panel B

foreach outcome of varlist ldonF donation_total_nrF hedgegroupF llobnewF  {
	qui reghdfe `outcome' PRisk_std lat, vce(cluster firm_id) ///
		absorb(i.sic2 i.cdateQ) keepsingletons
	est sto panelB_`outcome'
}

* c) Run regressions for panel C

local i = 0
foreach outcome in capital_investm_w_100 hiring_investm_w_100 ldonF llobnewF {

	local i = `i' + 1
	
	* Employment uses annual data
	if "`outcome'" == "hiring_investm_w_100" {
		preserve
			use ${firmyear_data}, clear
			qui reghdfe `outcome' PRisk_std PRisk_std_d2at d2at, vce(cluster firm_id) ///
				absorb(i.sic2 i.cdateY) keepsingletons
			est sto panelC_`i'
		restore
	}
	
	* All other variables use quarterly data
	else {
		qui reghdfe `outcome' PRisk_std PRisk_std_d2at d2at, vce(cluster firm_id) ///
			absorb(i.sic2 i.cdateQ) keepsingletons
		est sto panelC_`i'
	}
}

* d) Print output, panel by panel

esttab panelA_*, star(* 0.1 ** 0.05 *** 0.01) se
esttab panelB_*, star(* 0.1 ** 0.05 *** 0.01) se
esttab panelC_*, star(* 0.1 ** 0.05 *** 0.01) se

* =====================================================================
* 
* Reproduce Table 6: Mean vs. variance of political shocks 
*
* =====================================================================

est clear
use ${firmquarter_data}, clear

* a) Run regressions for all panels

local i = 0
foreach outcome in capital_investm_w_100 hiring_investm_w_100 llobnewF ldonF ///
	donation_total_nrF hedgegroupF {
	
	local i = `i' + 1
	
	* Employment uses annual data
	if "`outcome'" == "hiring_investm_w_100" {
		preserve
			use ${firmyear_data}, clear
			
			* 1) PRisk alone
			qui reghdfe `outcome' PRisk_std lat, vce(cluster firm_id) ///
				absorb(i.sic2 i.cdateY) keepsingletons
			est sto col1_`i'
			
			* 2) PRisk and PSentiment
			qui reghdfe `outcome' PRisk_std PSentiment_std lat, vce(cluster firm_id) ///
				absorb(i.sic2 i.cdateY) keepsingletons
			est sto col2_`i'
			
			* 3) PRisk, Sentiment
			qui reghdfe `outcome' PRisk_std Sentiment_std lat, vce(cluster firm_id) ///
				absorb(i.sic2 i.cdateY) keepsingletons
			est sto col3_`i'
			
			* 4) PRisk, PSentiment, stock returns, and earnings surprise
			qui reghdfe `outcome' PRisk_std av_retW_100 sue lat, vce(cluster firm_id) ///
				absorb(i.sic2 i.cdateY) keepsingletons
			est sto col4_`i'
		restore
	}
	
	* All other variables use quarterly data
	else {
		* 1) PRisk alone
		qui reghdfe `outcome' PRisk_std lat, vce(cluster firm_id) ///
			absorb(i.sic2 i.cdateQ) keepsingletons
		est sto col1_`i'
		
		* 2) PRisk and PSentiment
		qui reghdfe `outcome' PRisk_std PSentiment_std lat, vce(cluster firm_id) ///
			absorb(i.sic2 i.cdateQ) keepsingletons
		est sto col2_`i'
		
		* 3) PRisk, Sentiment
		qui reghdfe `outcome' PRisk_std Sentiment_std lat, vce(cluster firm_id) ///
			absorb(i.sic2 i.cdateQ) keepsingletons
		est sto col3_`i'
		
		* 4) PRisk, PSentiment, stock returns, and earnings surprise
		qui reghdfe `outcome' PRisk_std av_retW_100 sue lat, vce(cluster firm_id) ///
			absorb(i.sic2 i.cdateQ) keepsingletons
		est sto col4_`i'
	}
}

* b) Print output, panel by panel

esttab *_1 *_2, star(* 0.1 ** 0.05 *** 0.01) se
esttab *_3 *_4, star(* 0.1 ** 0.05 *** 0.01) se
esttab *_5 *_6, star(* 0.1 ** 0.05 *** 0.01) se

* =====================================================================
* 
* Reproduce Table 7: Falsification exercise: PRisk, NPRisk, and overall risk 
*
* =====================================================================

est clear
use ${firmquarter_data}, clear

* a) Run regressions for all panels

local i = 0
foreach outcome in capital_investm_w_100 hiring_investm_w_100 llobnewF ldonF ///
	donation_total_nrF hedgegroupF {
	
	local i = `i' + 1
	
	* Employment uses annual data
	if "`outcome'" == "hiring_investm_w_100" {
		preserve
			use ${firmyear_data}, clear
			
				* 1) PRisk alone
			qui reghdfe `outcome' PRisk_std PSentiment_std lat, vce(cluster firm_id) ///
				absorb(i.sic2 i.cdateY) keepsingletons
			est sto col1_`i'

			* 2) PRisk against NPRisk
			qui reghdfe `outcome' PRisk_std NPRisk_std PSentiment_std lat, vce(cluster firm_id) ///
				absorb(i.sic2 i.cdateY) keepsingletons
			est sto col2_`i'
			
			* 3) PRisk against Risk
			qui reghdfe `outcome' PRisk_std Risk_std PSentiment_std lat, vce(cluster firm_id) ///
				absorb(i.sic2 i.cdateY) keepsingletons
			est sto col3_`i'
		restore
	}
	
	* All other variables use quarterly data
	else {
		* 1) PRisk alone
		qui reghdfe `outcome' PRisk_std PSentiment_std lat, vce(cluster firm_id) ///
			absorb(i.sic2 i.cdateQ) keepsingletons
		est sto col1_`i'

		* 2) PRisk against NPRisk
		qui reghdfe `outcome' PRisk_std NPRisk_std PSentiment_std lat, vce(cluster firm_id) ///
			absorb(i.sic2 i.cdateQ) keepsingletons
		est sto col2_`i'
		
		* 3) PRisk against Risk
		qui reghdfe `outcome' PRisk_std Risk_std PSentiment_std lat, vce(cluster firm_id) ///
			absorb(i.sic2 i.cdateQ) keepsingletons
		est sto col3_`i'
	}
}

* b) Print output, panel by panel

esttab *_1 *_2, star(* 0.1 ** 0.05 *** 0.01) se
esttab *_3 *_4, star(* 0.1 ** 0.05 *** 0.01) se
esttab *_5 *_6, star(* 0.1 ** 0.05 *** 0.01) se

* =====================================================================
* 
* Reproduce Table 8: Variance decomposition of PRisk 
*
* =====================================================================

est clear
use ${firmquarter_data}, clear

* Open file handler and write first lines

capture file close ofile
file open ofile using "${table8_vardecomp}", write replace
file write ofile "\begin{table}\centering" _n
file write ofile "\begin{tabular}{l*{3}{r}}" _n
file write ofile "\hline\hline\addlinespace" _n
file write ofile "& (1) & (2) & (3) \\\hline\addlinespace" _n
file write ofile "Sector granularity & 2-digit SIC & 3-digit SIC "
file write ofile " & 4-digit SIC \\\addlinespace\hline\addlinespace" _n

* Create SIC3 and SIC4 id

tostring sic, gen(sic4)
replace sic4 = substr(sic4,1,4)
gen sic3 = substr(sic4,1,3)
egen sic4_id = group(sic4)
egen sic3_id = group(sic3)

* 1) Score on time effect

reghdfe PRisk_std if capital_investm_w_100 != ., absorb(i.cdateQ) keepsingletons
local r2_manual_1 = `e(r2)'
local n_manual_1 "`: di %9.0fc `e(N)''"

file write ofile "Time FE &"
file write ofile "`: di %9.2fc `=`r2_manual_1'*100''\% &"
file write ofile "`: di %9.2fc `=`r2_manual_1'*100''\% &"
file write ofile "`: di %9.2fc `=`r2_manual_1'*100''\% \\" _n

* 2) Score on sector effect

reghdfe PRisk_std if capital_investm_w_100 != ., ///
	absorb(i.cdateQ i.sic2) keepsingletons // sic2
quietly estadd scalar r2a `e(r2_a)'
local r2_manual_2 = `e(r2)'
local n_manual_2 "`: di %9.0fc `e(N)''"

reghdfe PRisk_std if capital_investm_w_100 != ., ///
	absorb(i.cdateQ i.sic3_id) keepsingletons // sic3
quietly estadd scalar r2a `e(r2_a)'
local r2_manual_2_sic3 = `e(r2)'
local n_manual_2_sic3 "`: di %9.0fc `e(N)''"

reghdfe PRisk_std if capital_investm_w_100 != ., ///
	absorb(i.cdateQ i.sic4_id) keepsingletons // sic4
quietly estadd scalar r2a `e(r2_a)'
local r2_manual_2_sic4 = `e(r2)'
local n_manual_2_sic4 "`: di %9.0fc `e(N)''"

file write ofile "Sector FE &"
file write ofile "`: di %9.2fc `=(`r2_manual_2'-`r2_manual_1')*100''\% &"
file write ofile "`: di %9.2fc `=(`r2_manual_2_sic3'-`r2_manual_1')*100''\% &"
file write ofile "`: di %9.2fc `=(`r2_manual_2_sic4'-`r2_manual_1')*100''\% \\" _n

* 3) Score on sector, sector*time effect

reghdfe PRisk_std if capital_investm_w_100 != ., ///
	absorb(i.sic2##i.cdateQ) keepsingletons // sic2
local r2_manual_3 = `e(r2)'
local n_manual_3 "`: di %9.0fc `e(N)''"

reghdfe PRisk_std if capital_investm_w_100 != ., ///
	absorb(i.sic3_id##i.cdateQ) keepsingletons // sic3
local r2_manual_3_sic3 = `e(r2)'
local n_manual_3_sic3 "`: di %9.0fc `e(N)''"

reghdfe PRisk_std if capital_investm_w_100 != ., ///
	absorb(i.sic4_id##i.cdateQ) keepsingletons // sic4
local r2_manual_3_sic4 = `e(r2)'
local n_manual_3_sic4 "`: di %9.0fc `e(N)''"

file write ofile "Sector \$\times\$ time FE &"
file write ofile "`: di %9.2fc `=(`r2_manual_3'-`r2_manual_2')*100''\% &"
file write ofile "`: di %9.2fc `=(`r2_manual_3_sic3'-`r2_manual_2_sic3')*100''\% &"
file write ofile "`: di %9.2fc `=(`r2_manual_3_sic4'-`r2_manual_2_sic4')*100''\%"
file write ofile "\\\addlinespace\hline\addlinespace" _n
file write ofile "\`\`\textbf{Firm-level}'' &"
file write ofile "`: di %9.2fc `=(1-`r2_manual_3')*100''\% &"
file write ofile "`: di %9.2fc `=(1-`r2_manual_3_sic3')*100''\% &"
file write ofile "`: di %9.2fc `=(1-`r2_manual_3_sic4')*100''\%"
file write ofile "\\\addlinespace\hline\addlinespace" _n

* 4) Score on sector*time, and firm effect

reghdfe PRisk_std if capital_investm_w_100 != ., ///
	absorb(i.firm i.sic2##i.cdateQ) keepsingletons // sic2
local r2_manual_4 = `e(r2)'
local n_manual_4 "`: di %9.0fc `e(N)''"

reghdfe PRisk_std if capital_investm_w_100 != ., ///
	absorb(i.firm i.sic3_id##i.cdateQ) keepsingletons // sic3
local r2_manual_4_sic3 = `e(r2)'
local n_manual_4_sic3 "`: di %9.0fc `e(N)''"

reghdfe PRisk_std if capital_investm_w_100 != ., ///
	absorb(i.firm i.sic4_id##i.cdateQ) keepsingletons // sic4
local r2_manual_4_sic4 = `e(r2)'
local n_manual_4_sic4 "`: di %9.0fc `e(N)''"
	
file write ofile "Permanent differences across firms within \\ sectors (Firm FE) &"
file write ofile "`: di %9.2fc `=(`r2_manual_4'-`r2_manual_3')*100''\% &"
file write ofile "`: di %9.2fc `=(`r2_manual_4_sic3'-`r2_manual_3_sic3')*100''\% &" _n
file write ofile "`: di %9.2fc `=(`r2_manual_4_sic4'-`r2_manual_3_sic4')*100''\% \\" _n
file write ofile "Variation over time in identity of firms within \\" _n
file write ofile "sectors most affected by political risk (residual) &"
file write ofile "`: di %9.2fc `=(1-`r2_manual_4')*100''\% &"
file write ofile "`: di %9.2fc `=(1-`r2_manual_4_sic3')*100''\% &"
file write ofile "`: di %9.2fc `=(1-`r2_manual_4_sic4')*100''\% "
file write ofile "\\\addlinespace\hline\addlinespace" _n
file write ofile "Number of sectors & 65 & 258 & 407 \\" _n
file write ofile "\addlinespace\hline\hline\addlinespace \end{tabular}" _n
file write ofile "\end{table}" _n
file close ofile

* =====================================================================
* 
* Reproduce Table 9: Measurement error
*
* =====================================================================

est clear
use ${firmyear_data}, clear

* Program to bootstrap noise to signal, IV

capture program drop noise_to_signalIV
program noise_to_signalIV, rclass

	version 14
	args y x1 x2 z1 fe1 fe2
	
	if "`fe1'" == "" {
		di "case: no fe"
		
		* OLS
		qui reg `y' `x1' `x2' if `z1' != .
		scalar beta_ols = _b[`x2']
		
		* IV
		qui ivreg2 `y' `x1' (`x2'=`z1')
		scalar beta_iv = _b[`x2']
	}
	else {
		di "case: fe"
		
		* OLS
		qui reghdfe `y' `x1' `x2' if `z1' != ., ///
			absorb(i.`fe1' i.`fe2' i.`fe1'#i.`fe2') old
		scalar beta_ols = _b[`x2']
		di e(command)
		di beta_ols
		
		* IV
		qui reghdfe `y' `x1' (`x2'=`z1'), ///
			absorb(i.`fe1' i.`fe2' i.`fe1'#i.`fe2') old
		scalar beta_iv = _b[`x2']
		di e(command)
		di beta_iv
	}
	
	* Noise-to-signal
	scalar ntos = (beta_iv/beta_ols) - 1

	* Share ME
	scalar shareme = 1 - (beta_ols/beta_iv)
	
	return scalar ntos = ntos
	return scalar shareme = shareme
end

* Program to bootstrap noise to signal, OLS

capture program drop noise_to_signalOLS
program noise_to_signalOLS, rclass

	version 14
	args y x1 x2 lag fe1 fe2
	
	if "`fe1'" == "" {
		
		* OLS
		qui reg `y' `x1' `x2' if `lag' != .
		scalar beta_ols = _b[`x2']
		
		* IV
		qui reg `y' `x1' `lag' if `x2' != .
		scalar beta_lag = _b[`lag']
	}
	else {
		
		* OLS
		qui reghdfe `y' `x1' `x2' if `lag' != ., ///
			absorb(i.`fe1' i.`fe2' i.`fe1'#i.`fe2') old
		scalar beta_ols = _b[`x2']
		
		* IV
		qui reghdfe `y' `x1' `lag' if `x2' != ., ///
			absorb(i.`fe1' i.`fe2' i.`fe1'#i.`fe2') old
		scalar beta_lag = _b[`lag']
	}
	
	* Noise-to-signal
	scalar ntos = (beta_lag/(beta_ols^2)) - 1

	* Share ME
	scalar shareme = 1 - ((beta_ols^2)/beta_lag)
	
	return scalar ntos = ntos
	return scalar shareme = shareme
end

* a) Run regressions for panel A

* OLS
qui reg PRisk_stdF PRisk_std lat if PRiskMDA_std != .
qui estadd local ntos = " "
qui estadd local sharem = " "
qui estadd local sharem_se = " "
qui estadd local test_sharem = " "
qui estadd local instr " "
est sto panelA_1

* Using PRiskMDA as instrument
qui reg PRisk_stdF PRisk_std lat if PRiskMDA_std != .
scalar beta_ols = _b[PRisk_std]
qui ivreg2 PRisk_stdF lat (PRisk_std=PRiskMDA_std)
scalar beta_iv = _b[PRisk_std]
qui estadd scalar ntos = (beta_iv/beta_ols) - 1
qui estadd scalar sharem = 1 - (beta_ols/beta_iv)
qui estadd local instr "PRisk10K"
est sto x
* bootstrap share_me=r(shareme), cluster(firm_id) seed(64827) reps(500): ///
*	noise_to_signalIV PRisk_stdF lat PRisk_std "PRiskMDA_std"
* local sharem_se = "(" + strtrim("`: di %12.3fc _se[share_me]'") + ")"
est resto x
qui estadd local sharem_se = "`sharem_se'"
est sto panelA_2

* Using L.PRiskMDA as instrument
qui reg PRisk_stdF PRisk_std lat if PRiskMDA_stdL != .
scalar beta_ols = _b[PRisk_std]
qui ivreg2 PRisk_stdF lat (PRisk_std=PRiskMDA_stdL)
scalar beta_iv = _b[PRisk_std]
qui estadd scalar ntos = (beta_iv/beta_ols) - 1
qui estadd scalar sharem = 1 - (beta_ols/beta_iv)
qui estadd local instr "L.PRisk10K"
est sto x
* bootstrap share_me=r(shareme), cluster(firm_id) seed(64827) reps(500): ///
*	noise_to_signalIV PRisk_stdF lat PRisk_std "PRiskMDA_stdL"
* local sharem_se = "(" + strtrim("`: di %12.3fc _se[share_me]'") + ")"
est resto x
qui estadd local sharem_se = "`sharem_se'"
est sto panelA_3

* Using L.PRisk
qui reg PRisk_stdF PRisk_std lat if PRisk_stdL != .
scalar beta_ols = _b[PRisk_std]
qui ivreg2 PRisk_stdF lat (PRisk_std=PRisk_stdL)
scalar beta_iv = _b[PRisk_std]
qui estadd scalar ntos = (beta_iv/beta_ols) - 1
qui estadd scalar sharem = 1 - (beta_ols/beta_iv)
qui estadd local instr "L.PRisk"
est sto x
* bootstrap share_me=r(shareme), cluster(firm_id) seed(64827) reps(500): ///
*	noise_to_signalIV PRisk_stdF lat PRisk_std "PRisk_stdL"
* local sharem_se = "(" + strtrim("`: di %12.3fc _se[share_me]'") + ")"
est resto x
qui estadd local sharem_se = "`sharem_se'"
est sto panelA_4

* b) Run regressions for panel B

* OLS
qui reghdfe PRisk_stdF PRisk_std lat if PRiskMDA_std != ., ///
	absorb(i.cdateY i.sic2 i.cdateY#i.sic2)
qui estadd local ntos = " "
qui estadd local sharem = " "
qui estadd local sharem_se = " "
qui estadd local test_sharem = " "
qui estadd local instr " "
est sto panelB_1

* Using PRiskMDA as instrument
qui reghdfe PRisk_stdF PRisk_std lat if PRiskMDA_std !=., ///
	absorb(i.cdateY i.sic2 i.cdateY#i.sic2)
scalar beta_ols = _b[PRisk_std]
qui reghdfe PRisk_stdF lat (PRisk_std=PRiskMDA_std), ///
	absorb(i.cdateY i.sic2 i.cdateY#i.sic2) old
scalar beta_iv = _b[PRisk_std]
qui estadd scalar ntos = (beta_iv/beta_ols) - 1
qui estadd scalar sharem = 1 - (beta_ols/beta_iv)
qui estadd local instr "PRisk10K"
est sto x
* bootstrap share_me=r(shareme), cluster(firm_id) seed(64827) reps(500): ///
*	noise_to_signalIV PRisk_stdF lat PRisk_std "PRiskMDA_std" cdateY sic2
* local sharem_se = "(" + strtrim("`: di %12.3fc _se[share_me]'") + ")"
est resto x
qui estadd local sharem_se = "`sharem_se'"
est sto panelB_2

* Using L.PRiskMDA as instrument
qui reghdfe PRisk_stdF PRisk_std lat if PRiskMDA_stdL != ., ///
	absorb(i.cdateY i.sic2 i.cdateY#i.sic2)
scalar beta_ols = _b[PRisk_std]
qui reghdfe PRisk_stdF lat (PRisk_std=PRiskMDA_stdL), ///
	absorb(i.cdateY i.sic2 i.cdateY#i.sic2) old
scalar beta_iv = _b[PRisk_std]
qui estadd scalar ntos = (beta_iv/beta_ols) - 1
qui estadd scalar sharem = 1 - (beta_ols/beta_iv)
qui estadd local instr "L.PRisk10K"
est sto x
* bootstrap share_me=r(shareme), cluster(firm_id) seed(64827) reps(500): ///
*	noise_to_signalIV PRisk_stdF lat PRisk_std "PRiskMDA_stdL" cdateY sic2
* local sharem_se = "(" + strtrim("`: di %12.3fc _se[share_me]'") + ")"
est resto x
qui estadd local sharem_se = "`sharem_se'"
est sto panelB_3

* Using L.PRisk
qui reghdfe PRisk_stdF PRisk_std lat if PRisk_stdL != ., absorb(i.cdateY i.sic2 i.cdateY#i.sic2) old
scalar beta_ols = _b[PRisk_std]
qui reghdfe PRisk_stdF lat (PRisk_std=PRisk_stdL), absorb(i.cdateY i.sic2 i.cdateY#i.sic2) old
scalar beta_iv = _b[PRisk_std]
qui estadd scalar ntos = (beta_iv/beta_ols) - 1
qui estadd scalar sharem = 1 - (beta_ols/beta_iv)
qui estadd local instr "L.PRisk"
est sto x
* bootstrap share_me=r(shareme), cluster(firm_id) seed(64827) reps(500): ///
*	noise_to_signalIV PRisk_stdF lat PRisk_std "PRisk_stdL" cdateY sic2
* local sharem_se = "(" + strtrim("`: di %12.3fc _se[share_me]'") + ")"
est resto x
qui estadd local sharem_se = "`sharem_se'"
est sto panelB_4

* c) Print output, panel by panel

esttab panelA_*, star(* 0.1 ** 0.05 *** 0.01) se scalars(instr N sharem ///
	sharem_se) sfmt(%7.2g %9.0fc %9.3f %7.2g)
esttab panelB_*, star(* 0.1 ** 0.05 *** 0.01) se scalars(instr N sharem ///
	sharem_se) sfmt(%7.2g %9.0fc %9.3f %7.2g)

* =====================================================================
* 
* Reproduce Table 10: The nature of firm-level political risk
*
* =====================================================================

est clear
use ${firmquarter_data}, clear

* a) Run regressions for all panels

foreach outcome of varlist impvol_w_std volatility_w_std {

	*1) Baseline
	qui reghdfe `outcome' PRisk_std lat, vce(cluster firm_id) ///
		absorb(i.cdateQ i.sic2 i.sic2#i.cdateQ) keepsingletons
	est sto `outcome'_1
		
	*2) Interactions between betas generated from a regression of PRisk_{it} on PRisk_t and mean risk
	qui reghdfe `outcome' PRisk_std beta_PRisk_aPRisk_std lat, vce(cluster firm_id) ///
		absorb(i.sic2 i.cdateQ i.sic2#i.cdateQ) keepsingletons
	est sto `outcome'_2

	*3) Interactions between time-varying betas generated from a regression of PRisk_{it} on PRisk_t and mean risk
	qui reghdfe `outcome' PRisk_std beta2_PRisk_aPRisk_std lat, vce(cluster firm_id) ///
		absorb(i.sic2 i.cdateQ i.sic2#i.cdateQ) keepsingletons
	est sto `outcome'_3

	*2) Interactions between EPU beta and mean risk
	qui reghdfe `outcome' PRisk_std beta_aPRisk_std lat, vce(cluster firm_id) ///
		absorb(i.sic2 i.cdateQ i.sic2#i.cdateQ) keepsingletons
	est sto `outcome'_4

	*3) Interactions between time-varying EPU beta and mean risk
	qui reghdfe `outcome' PRisk_std beta2_aPRisk_std lat, vce(cluster firm_id) ///
		absorb(i.sic2 i.cdateQ i.sic2#i.cdateQ) keepsingletons
	est sto `outcome'_5

	*4) Controling for federal contracts
	qui reghdfe `outcome' PRisk_std lat lcontractamount, vce(cluster firm_id) ///
		absorb(i.cdateQ i.sic2 i.sic2#i.cdateQ) keepsingletons
	est sto `outcome'_6

	*5) Controling for federal contracts and interaction
	qui reghdfe `outcome' PRisk_std lat lcontractamount lcontractamount_aPRisk_std, ///
		vce(cluster firm_id) absorb(i.cdateQ i.sic2 i.sic2#i.cdateQ) keepsingletons
	est sto `outcome'_7
}

* b) Print output, panel by panel

esttab impvol_w_std_*, star(* 0.1 ** 0.05 *** 0.01) se scalars(instr N sharem ///
	sharem_se) sfmt(%7.2g %9.0fc %9.3f %7.2g)
esttab volatility_w_std_*, star(* 0.1 ** 0.05 *** 0.01) se scalars(instr N sharem ///
	sharem_se) sfmt(%7.2g %9.0fc %9.3f %7.2g)

* =====================================================================
* 
* Reproduce Table 11: Topic-specific lobbying and topic-specific political risk
*
* =====================================================================

est clear
use ${firmqtopicquarter_data}, clear

* a) Run regressions for all panels

foreach outcome of varlist topic_dummyF_100 llobnewF {

	* Sector and time fixed effects
	qui reghdfe `outcome' PRiskT_std lat, vce(cluster firm_id) ///
		absorb(i.sic2 i.cdateQ) keepsingletons
	est sto `outcome'_1

	* Topic, time, and sector fixed effects
	qui reghdfe `outcome' PRiskT_std lat, vce(cluster firm_id) ///
		absorb(i.topic_id i.sic2 i.cdateQ) keepsingletons
	est sto `outcome'_2

	* Firm, time, and topic fixed effects
	qui reghdfe `outcome' PRiskT_std lat, vce(cluster firm_id) ///
		absorb(i.firm_id i.cdateQ i.topic_id) keepsingletons
	est sto `outcome'_3

	* Firm, industry*time and topic fixed effects
	qui reghdfe `outcome' PRiskT_std lat, vce(cluster firm_id) ///
		absorb(i.firm_id i.cdateQ#i.sic2 i.topic_id i.cdateQ) keepsingletons
	est sto `outcome'_4

	* Firm*topic, time fixed effects
	qui reghdfe `outcome' PRiskT_std lat, vce(cluster firm_id) ///
		absorb(i.firm_id##i.topic_id i.cdateQ##i.sic2) keepsingletons
	est sto `outcome'_5
}

* b) Print output, panel by panel

esttab topic_dummyF_100_*, star(* 0.1 ** 0.05 *** 0.01) se scalars(instr N sharem ///
	sharem_se) sfmt(%7.2g %9.0fc %9.3f %7.2g)
esttab llobnewF_*, star(* 0.1 ** 0.05 *** 0.01) se scalars(instr N sharem ///
	sharem_se) sfmt(%7.2g %9.0fc %9.3f %7.2g)

* =====================================================================
* 
* Reproduce Table 12: Case studies: Obama-era budget crises 
*
* =====================================================================

est clear
use ${firmqtopicquarter_data}, clear

* Generate differenced outcome
egen firmtopic_id = group(gvkey topic_id)
xtset firmtopic_id cdateQ
gen dPRiskT_std = D.PRiskT_std

* Generate sum of bigram counts for all three events
gen allthree = instrument_debtceiling + instrument_govshutdown + instrument_fiscalcliff

* Generate polynomials
gen instr_debtceiling2 = instrument_debtceiling ^ 2
gen instr_debtceiling3 = instrument_debtceiling ^ 3
gen instr_fiscalcliff2 = instrument_fiscalcliff ^ 2
gen instr_fiscalcliff3 = instrument_fiscalcliff ^ 3
gen instr_govshutdown2 = instrument_govshutdown ^ 2
gen instr_govshutdown3 = instrument_govshutdown ^ 3
gen allthree2 = allthree ^ 2
gen allthree3 = allthree ^ 3

* a) Run regressions for panel A

* Column 1: Debt ceiling alone
qui reg dPRiskT_std instrument_debtceiling lat ///
	if topic_id == 2 & cdateQ == 206, vce(robust)
qui estadd local periods "2011-q3"
qui est sto panelA_1

* Column 2: Debt ceiling with fiscal cliff
qui reg dPRiskT_std instrument_debtceiling instrument_fiscalcliff lat ///
	if topic_id == 2 & cdateQ == 212, vce(robust)
qui estadd local periods "2013-q1"
qui est sto panelA_2

* Column 3: Debt ceiling with government shutdown
qui reg dPRiskT_std instrument_debtceiling instrument_govshutdown lat ///
	if topic_id == 2 & cdateQ == 215, vce(robust)
qui estadd local periods "2013-q4"
qui est sto panelA_3

* Column 4: All three on entire sample
qui reghdfe PRiskT_std allthree lat if topic_id == 2, vce(cluster firm_id) ///
	absorb(cdateQ i.firm_id i.sic2##i.cdateQ) keepsingletons
qui estadd local periods "All"
qui est sto panelA_4

* b) Run regressions for panel B

* Column 1: Lobbying as outcome
qui reghdfe topic_dummyF_100 allthree lat if topic_id == 2, ///
	vce(cluster firm_id) absorb(cdateQ i.firm_id i.sic2##i.cdateQ) keepsingletons
qui estadd local model_spec "OLS"
qui est sto panelB_1

* Column 2: OLS with lobbying as outcome
qui reghdfe topic_dummyF_100 PRiskT_std lat if topic_id == 2, ///
	vce(cluster firm_id) absorb(cdateQ i.firm_id i.sic2##i.cdateQ) keepsingletons
qui estadd local model_spec "OLS"
qui est sto panelB_2

* Column 3: IV with lobbying as outcome
qui reghdfe topic_dummyF_100 lat (PRiskT_std=allthree allthree2 allthree3 ///
	instrument_debtceiling instr_debtceiling2 instr_debtceiling3 ///
	instrument_fiscalcliff instr_fiscalcliff2 instr_fiscalcliff3 ///
	instrument_govshutdown instr_govshutdown2 instr_govshutdown3) ///
	if topic_id == 2, vce(cluster firm_id) ///
	absorb(cdateQ i.firm_id i.sic2##i.cdateQ) ffirst keepsingletons old
qui estadd scalar fs_f = e(cdf)
qui estadd local model_spec "IV"
qui est sto panelB_3

*Column 4: IV with lobbying ($) as outcome
qui reghdfe llobnewF lat (PRiskT_std=allthree allthree2 allthree3 ///
	instrument_debtceiling instr_debtceiling2 instr_debtceiling3 ///
	instrument_fiscalcliff instr_fiscalcliff2 instr_fiscalcliff3 ///
	instrument_govshutdown instr_govshutdown2 instr_govshutdown3) ///
	if topic_id == 2, vce(cluster firm_id) ///
	absorb(cdateQ i.firm_id i.sic2##i.cdateQ) ffirst keepsingletons old
qui estadd scalar fs_f = e(cdf)
qui estadd local model_spec "IV"
qui est sto panelB_4

* c) Print output, panel by panel

esttab panelA_*, star(* 0.1 ** 0.05 *** 0.01) se label scalars(periods)
esttab panelB_*, star(* 0.1 ** 0.05 *** 0.01) se label scalars(model_spec)

* End
