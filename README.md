# MSSP-Public-Use-Files
# MSSP EM correlation to Risk Scores

# This python script pulls 2022 and 2021 MSSP Financial and Quality Results from the CMS API, then prompts for the 2020 filepath as a .csv
# CMS as of 11/27/2023 is reconfiguring the 2013 - 2020 public use files and they are not currently available for download, so the user needs to have previously saved the 2020 file for this to run appropriately.
# Script then outputs 3 plots evaluating the correlation between change in primary care visits from 2020 to 2021 and the Aged / Non-Dual HCC risk score change from 2021 to 2022.
# Results appear to suggest there is no positive correlation between these 2 variables, and if anything, very slight negative correlation.
# Total primary care E&M and both the subset of PCP and Specialist primary care E&M are evaluated.
