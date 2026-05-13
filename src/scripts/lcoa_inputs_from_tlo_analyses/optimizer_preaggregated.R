# Standalone preaggregated optimizer for Python integration.
# - ce_dalys, conscost, and hr_* are preaggregated totals at full implementation.
# - Decision variables represent fractions of each intervention implemented.
# - feascov and substitute/compulsory constraints still bound implementation shares.

library(lpSolve)
library(jsonlite)

find_optimal_package <- function(inputs, objective_input, cet_input,
                                 drug_budget_input, drug_budget.scale,
                                 hr.time.constraint, hr.size, hr.scale,
                                 use_feasiblecov_constraint, feascov_scale, compcov_scale,
                                 compulsory_interventions, substitutes, task_shifting_pharm) { # % complements %

  ## Total DALYs averted based on CE evidence; this was per person in the original
  ## script but is cumulative in this version
  dalys <- as.numeric(as.character(inputs$ce_dalys))
  ## Cumulative cost of drugs and commodities
  drugcost <- as.numeric(as.character(inputs$conscost))
  maxcoverage <- 1 ## SB: Changed as I am not using use_feasiblecov_constraint
  ## Preaggregated mode: unit case scaling
  cases <- rep(1, length(dalys))
  ## Full cost
  fullcost <- as.numeric(as.character(inputs$ce_cost))
  ## Number of minutes of health worker time required per intervention
  hrneed <-
    inputs[c("hr_clin", "hr_nur", "hr_pharm", "hr_lab", "hr_ment", "hr_nutri")]
  hrneed <- as.data.frame(apply(hrneed, 2, as.numeric))

  n <- length(dalys) # number of interventions included in the analysis

  ###################################
  # 3.1 Set up LPP
  ###################################

  # Objective - maximize DALYs
  #****************************************************
  # Define net health
  cet <- cet_input
  nethealth <- dalys - fullcost / cet

  # Define objective
  if (objective_input == "nethealth") {
    objective <- nethealth * cases
  } else if (objective_input == "dalys") {
    objective <- dalys * cases
  } else {
    print("ERROR: objective_input can take values dalys or nethealth")
  }

  # Constraints - 1. Drug Budget, 2. HR Requirements
  #****************************************************
  # 1. Drug Budget
  #----------------
  cons_drug <- drugcost * cases # Cost of drugs for the number of cases covered
  cons_drug.limit <- drug_budget_input * drug_budget.scale
  cons_drug.limit_base <- drug_budget_input # unscaled drug budget

  # 2. HR Constraints
                                        #---------------------
  ## HR minutes required to deliver intervention to all cases in need
  hr_minutes_need <- hrneed * cases[row(hrneed)]

  ## Update HR constraints so that nurses, pharmacists, medical officers, etc. represent joint constraints
  ## Medical officer + Clinical officer + Medical Assistant
  clinicalstaff.need <- hr_minutes_need[c("hr_clin")]
  ## Nurse officer + Nurse midwife
  nursingstaff.need <- hr_minutes_need[c("hr_nur")]
  ## Pharmacist + Pharmacist Technician + Pharmacist Assistant
  pharmstaff.need <- hr_minutes_need[c("hr_pharm")]
  ## Lab officer + Lab technician + Lab assistant
  labstaff.need <- hr_minutes_need[c("hr_lab")]
  # remove CHW
  mentalstaff.need <- hr_minutes_need[c("hr_ment")] # Mental health staff
  nutristaff.need <- hr_minutes_need[c("hr_nutri")] # Nutrition staff

  # Clean total minutes available per cadre
  cons_hr.limit <- hr.time.constraint
  clinicalstaffmins.limit <- cons_hr.limit[1]
  nursingstaffmins.limit <- cons_hr.limit[2]
  pharmstaffmins.limit <- cons_hr.limit[3]
  labstaffmins.limit <- cons_hr.limit[4]
  mentalstaffmins.limit <- cons_hr.limit[5]
  nutristaffmins.limit <- cons_hr.limit[6]

  reps <- 4 # set the number of times that the matrix of interventions is duplicated

  # Define a function which duplicates a matrix horizontally
  duplicate_matrix_horizontally <- function(reps, matrix) {
    matrix <- do.call(rbind, replicate(reps, matrix, simplify = FALSE))
  }

  if (task_shifting_pharm == 0) {
    print("")
  } else if (task_shifting_pharm == 1) {
    clinicalstaff.need <-
      duplicate_matrix_horizontally(reps, as.matrix(clinicalstaff.need))
    nursingstaff.need <-
      rbind(as.matrix(nursingstaff.need),
            as.matrix(nursingstaff.need + pharmstaff.need),
            as.matrix(nursingstaff.need + nutristaff.need),
            as.matrix(nursingstaff.need + nutristaff.need + pharmstaff.need))
    pharmstaff.need <-
      rbind(as.matrix(pharmstaff.need), as.matrix(rep(0, n)),
            as.matrix(pharmstaff.need), as.matrix(rep(0, n)))
    labstaff.need <-
      duplicate_matrix_horizontally(reps, as.matrix(labstaff.need))
    mentalstaff.need <-
      duplicate_matrix_horizontally(reps, as.matrix(mentalstaff.need))
    nutristaff.need <-
      rbind(as.matrix(nutristaff.need), as.matrix(nutristaff.need),
            as.matrix(rep(0, n)), as.matrix(rep(0, n)))
  } else {
    print("ERROR: tash_shifting_pharm can take values 0 or 1")
  }

  # Clean total workforce size per cadre
  hr_size.limit <- hr.size
  clinicalstaff.limit <- hr_size.limit[1]
  nursingstaff.limit <- hr_size.limit[2]
  pharmstaff.limit <- hr_size.limit[3]
  labstaff.limit <- hr_size.limit[4]
  mentalstaff.limit <- hr_size.limit[5]
  nutristaff.limit <- hr_size.limit[6]

  clinicalstaff.scale <- hr.scale[1]
  nursestaff.scale <- hr.scale[2]
  pharmstaff.scale <- hr.scale[3]
  labstaff.scale <- hr.scale[4]
  mentalstaff.scale <- hr.scale[5]
  nutristaff.scale <- hr.scale[6]

  ## Each list here represents the number of staff (of each cadre) needed to deliver each intervention to all cases in need.
  ## Eg. for each cesarean section, 45 minutes of medical staff's time is needed (or 104,200 minutes for 2316 cases). On average 39,900 minutes are available per medical staff each year (257.3 million minutes in total divided by 6,400 medical staff). This means that for 2136 cases, 2.16 medical staff are needed (2316*45/(257.3m/6400))
  cons_hr <-
    cbind(clinicalstaff.need / (clinicalstaffmins.limit / clinicalstaff.limit),
          nursingstaff.need / (nursingstaffmins.limit / nursingstaff.limit),
          pharmstaff.need / (pharmstaffmins.limit / pharmstaff.limit),
          labstaff.need / (labstaffmins.limit / labstaff.limit),
          mentalstaff.need / (mentalstaffmins.limit / mentalstaff.limit),
          nutristaff.need / (nutristaffmins.limit / nutristaff.limit))
  cons_hr.saved <- cons_hr

  cons_hr.limit_base <-
    cbind(clinicalstaff.limit, nursingstaff.limit, pharmstaff.limit,
          labstaff.limit, mentalstaff.limit, nutristaff.limit)
  cons_hr.limit <-
    cbind(clinicalstaff.limit * clinicalstaff.scale,
          nursingstaff.limit * nursestaff.scale,
          pharmstaff.limit * pharmstaff.scale,
          labstaff.limit * labstaff.scale,
          mentalstaff.limit * mentalstaff.scale,
          nutristaff.limit * nutristaff.scale)

  colnames(cons_hr.limit) <- colnames(cons_hr)
  cons_hr.limit.saved <- cons_hr.limit

  # Combine the constraints into one matrix
  #****************************************************
  # 1. HR
  #--------------------------------------
  cons_hr <- as.matrix(cons_hr)
  cons_hr.limit <- as.matrix(cons_hr.limit)

  # 2. Drug
  #--------------------------------------
  cons_drug <- as.matrix(cons_drug)
  cons_drug.limit <- as.matrix(cons_drug.limit)

  # 3. Max coverage
  #--------------------------------------
  cons.feascov <- diag(x = cases, n, n)
  if (use_feasiblecov_constraint == 1) {
    cons.feascov.limit <- as.matrix(maxcoverage * feascov_scale * cases)
  } else if (use_feasiblecov_constraint == 0) {
    cons.feascov.limit <- as.matrix(cases) # changed the constraint on 12May (multiplied by cases)
  } else {
    print("ERROR: use_feasiblecov_constraint can take values 0 or 1")
  }

  nonneg.lim <- as.matrix(rep(0, n))

  # 4. Compulsory interventions
  #--------------------------------------
  if (length(compulsory_interventions) > 0) {
    comp.count <- length(compulsory_interventions)
    cons_compulsory <- matrix(0L, length(compulsory_interventions), ncol = n)
    cons_compulsory.limit <- matrix(0L, length(compulsory_interventions), ncol = 1)
    for (i in 1:length(compulsory_interventions)) {
      a <- which(inputs$intcode == compulsory_interventions[i])
      b <- inputs$intervention[a]
      # print(paste("Compulsory intervention: ",b, "; Code: ", compulsory_interventions[i], "; Number ",a ))
      cons_compulsory[i, a] <- cases[a]
      # CHECK THIS CHANGE MADE on 26Aug21
      cons_compulsory.limit[i] <- cases[a] * maxcoverage[a] * feascov_scale * compcov_scale # changed on 12May to maxcoverage because cons.feascov.limit is now maximum number of cases rather than maximum % coverage
    }
    dim(cons_compulsory)
  } else if (length(compulsory_interventions) == 0) {
    comp.count <- 1
    cons_compulsory <- matrix(0L, 1, ncol = n)
    cons_compulsory.limit <- matrix(0L, 1, ncol = 1)
  }
  cons_compulsory <- t(cons_compulsory)

  # placeholder#
  ###### % Complementary interventions code left out for now %

  # 5. Substitute interventions
  #--------------------------------------
  substitutes <- substitutes
  subs.count <- length(substitutes)
  cons_substitutes.limit <- matrix(0L, length(substitutes), ncol = 1)
  cons_substitutes <- matrix(0L, length(substitutes), ncol = n)

  # First find the maximum number of feasible cases among the substitute interventions
  subsgrp_casesmax <- matrix(0L, length(substitutes), ncol = 1)
  for (i in 1:subs.count) {
    for (j in substitutes[i]) {
      subsgrp_cases <- 0
      for (k in j) {
        a <- which(inputs$intcode == k)
        if (use_feasiblecov_constraint == 1) {
          cases_max <- cases[a] * maxcoverage[a] * feascov_scale
        } else if (use_feasiblecov_constraint == 0) {
          cases_max <- cases[a]
        }
        subsgrp_cases <- cbind(subsgrp_cases, cases_max)
      }
      subsgrp_casesmax[i] <- max(subsgrp_cases)
      # print(paste("Group", i, "Cases max", subsgrp_casesmax[i]))
    }
  }

  # Next define the constraint such that the sum of the cases for each substitute interventions is less than or equal to the maxumum feasible cases derived above
  # print("Substitutes")
  for (i in 1:subs.count) {
    # print(paste("Substitute group", i))
    # print("------------------------------------------------------------")
    for (j in substitutes[i]) {
      for (k in j) {
        a <- which(inputs$intcode == k)
        b <- inputs$intervention[a]
        #     print(paste("Intervention: ",b, "; Code: ", k, "; Maximum cases for intervention:", cons.feascov.limit[a],"; Number: ",a))
        cons_substitutes[i, a] <- cases[a] # changed on 12May from 1 to cases
        cons_substitutes.limit[i] <- subsgrp_casesmax[i] # changed on 12May to maxcoverage because cons.feascov.limit is now maximum number of cases rather than maximum % coverage
      }
    }
    # cons_substitutes.limit[i] <- cons_substitutes.limit[i]/lengths(substitutes)[i]  # removed on 12May
    # print(paste("Maximum combined cases for group ",i, "= ", subsgrp_casesmax[i])) # print suppressed
  }
  cons_substitutes <- t(cons_substitutes)

  # Changes to constraints if task-shifting of pharmacist responsibility is allowed
  #--------------------------------------------------------------------------------
  # Update the constraint matrices if task shifting is allowed
  if (task_shifting_pharm == 0) {
    print("No task shifting of pharmaceutical tasks")
  } else if (task_shifting_pharm == 1) {
    # 1. Objective
    objective <- duplicate_matrix_horizontally(reps, as.matrix(objective))
    # 2. Drug budget constraint (cons_drug.limit does not need to be changed)
    cons_drug <- duplicate_matrix_horizontally(reps, as.matrix(cons_drug))
    # 3. Feasible coverage constraint
    cons.feascov <- duplicate_matrix_horizontally(reps, as.matrix(cons.feascov))
    # 4. Compulsory interventions
    cons_compulsory <- duplicate_matrix_horizontally(reps, as.matrix(cons_compulsory))
    # 6. Substitutes
    cons_substitutes <- duplicate_matrix_horizontally(reps, as.matrix(cons_substitutes))
  } else {
    print("ERROR: task_shifting_pharm can take values 0 or 1")
  }

  # Combine constraints 1-5
  print(dim(t(cons_drug)))
  print(dim(t(cons_hr)))
  print(dim(t(cons.feascov)))
  print(dim(t(cons_compulsory)))
  print(dim(t(cons_substitutes)))
  cons.mat <- rbind(t(cons_drug), t(cons_hr), t(cons.feascov), t(cons.feascov), t(cons_compulsory), t(cons_substitutes)) # % cons_complements %
  dim(cons.mat)
  cons.mat.limit <- rbind(cons_drug.limit, t(cons_hr.limit), cons.feascov.limit, nonneg.lim, cons_compulsory.limit, cons_substitutes.limit) # cons_complements.limit,
  dim(cons.mat.limit)
  print(dim(cons.mat))
  print(dim(cons.mat.limit))

  # Direction of relationship
  cons.dir <- rep("<=", 1 + 8 + n)
  cons.dir <- c(cons.dir, rep(">=", n), rep(">=", comp.count))
  cons.dir <- c(cons.dir, rep("<=", length(substitutes)))
  # % cons.dir <- c(cons.dir,rep("<=",length(complements))) %
  length(cons.dir)
  length(cons.dir) <- dim(cons.mat.limit)[1] # Assert that the length of the directions list is the same as that of the constraints matrix

  ###################################
  # 3.2 - Run LPP
  ###################################

  solution.class <- lp("max", objective, cons.mat, cons.dir, cons.mat.limit, compute.sens = TRUE)
  print(solution.class$status) # 0 means optimal in lpSolve
  print(solution.class$objval)

  saveRDS(solution.class, file = "solution.rds")
  ###################################
  # 3.3 - Outputs
  ###################################
  # Export solution to a .csv file
  #------------------------------------
    solution <- as.data.frame(solution.class$solution)

  solution_hr <- as.data.frame(solution.class$solution) # use this uncollapsed version of the dataframe for HR use calculations below
  # Collapse solution by intervention
  if (task_shifting_pharm == 1) {
    for (i in 1:length(dalys)) {
      for (j in 1:(reps - 1)) {
        solution[i, 1] <- solution[i, 1] + solution[i + length(dalys) * j, 1]
      }
    }
    solution <- as.data.frame(solution[1:length(dalys), 1])
  }

  # Number of interventions with a positive net health impact
  pos_nethealth.count <- sum(nethealth > 0) # this seems to be one less than the figure in the excel

  # Number of interventions in the optimal package
  intervention.count <- sum(solution != 0)

  # DALY burden averted as a % of avertible DALY burden
  solution_dalysaverted <- solution * cases * dalys # Dalys averted per intervention
  dalysavertible <- cases * dalys # Total DALYs that can be averted at maximum coverage
  dalys_averted <- round(sum(unlist(lapply(solution_dalysaverted, sum))), 2)
  dalys_averted.prop <- sum(unlist(lapply(solution_dalysaverted, sum))) / sum(unlist(lapply(dalysavertible, sum)))

  # Drugs and Commodities cost (% of budget available)
  solution_drugexp <- solution * cons_drug[1:length(dalys), ] # Total drug budget required per intervention for the  the optimal solution
  total_drug_exp <- round(sum(unlist(lapply(solution_drugexp, sum))), 2) # Total drug budget required for the  the optimal solution
  drug_exp.prop <- total_drug_exp / cons_drug.limit_base

  # Total HR use (% of capacity)
  hr_cadres <- c("Clinical staff", "Nurse", "Pharmacist", "Lab", "Mental", "Nutrition")
  solution_hruse <- unlist(solution_hr) * cons_hr # Number of minutes per health worker cadre and intervention utlitised by the optimal solution
  if (task_shifting_pharm == 1) {
    for (i in 1:length(dalys)) {
      for (j in 1:(reps - 1)) {
        solution_hruse[i, ] <- solution_hruse[i, ] + solution_hruse[i + length(dalys) * j, ]
      }
    }
    solution_hruse <- solution_hruse[1:length(dalys), ]
  }
  total_hruse <- colSums(solution_hruse, na.rm = FALSE, dims = 1) # Number of minutes per health worker cadre utlitised by the optimal solution
  hruse.prop <- round(total_hruse / cons_hr.limit_base, 2)
  colnames(hruse.prop) <- hr_cadres

  # Cost-effectiveness Threshold
  icer <- fullcost / dalys
  temp <- cbind.data.frame(icer, solution, inputs$intervention)
  temp["solution.class$solution"] <- as.numeric(temp[[2]])
  temp["icer"] <- as.numeric(temp[[1]])
  cet_soln <- round(max(temp["icer"][temp["solution.class$solution"] > 0]), 2) # previoiusly temp$icer[temp$solution > 0]
  a <- which(icer == max(temp["icer"][temp["solution.class$solution"] > 0])) # to check which included intervention has the highest ICER
  least.ce.intervention <- inputs$intervention[a]

  # Collapse above outputs so that each intervention appears once in the list irrespective of task-shifting
  # pos_nethealth.count, intervention.count, dalys_averted, cet_soln, drug_exp.prop, t(hruse.prop[,visible_cadres])

  list(
    "Total number of interventions in consideration" = length(dalys),
    "Number of interventions with positive net health impact" = pos_nethealth.count,
    "Number of interventions in the optimal package" = intervention.count,
    "Net DALYs averted" = solution.class$objval,
    "Total DALYs averted" = sum(unlist(lapply(solution_dalysaverted, sum))),
    "Proportion of DALY burden averted" = dalys_averted.prop,
    "Proportion of drug budget used" = drug_exp.prop,
    "Proportion of HR capacity used by cadre" = hruse.prop,
    "CET based on solution" = cet_soln
  )
}


run_optimizer_from_csv <- function() {
  inputs <- readr::read_csv("optimizer_inputs.csv", show_col_types = TRUE)
  constraints <- readr::read_csv("hr_constraints.csv", show_col_types = TRUE )
  hr.time.constraint <- c(
    constraints$capacity[constraints$Officer_Category == "hr_clin"],
    constraints$capacity[constraints$Officer_Category == "hr_nur"],
    constraints$capacity[constraints$Officer_Category == "hr_pharm"],
    constraints$capacity[constraints$Officer_Category == "hr_lab"],
    constraints$capacity[constraints$Officer_Category == "hr_ment"],
    constraints$capacity[constraints$Officer_Category == "hr_nutri"]
  )

  hr.size.constraint <- c(
      constraints$staff_count[constraints$Officer_Category == "hr_clin"],
      constraints$staff_count[constraints$Officer_Category == "hr_nur"],
      constraints$staff_count[constraints$Officer_Category == "hr_pharm"],
      constraints$staff_count[constraints$Officer_Category == "hr_lab"],
      constraints$staff_count[constraints$Officer_Category == "hr_ment"],
      constraints$staff_count[constraints$Officer_Category == "hr_nutri"]
  )


  res <- find_optimal_package(
    inputs = inputs,
    objective_input = "dalys",
    cet_input = 600,
    drug_budget_input = 225602946, #TODO - get this from constaint
    drug_budget.scale = 1,
    hr.time.constraint = hr.time.constraint,
    hr.size = hr.size.constraint,
    hr.scale = rep(1, length(hr.time.constraint)),
    use_feasiblecov_constraint = 0,
    feascov_scale = 1,
    compcov_scale = 1,
    compulsory_interventions = c(),
    substitutes = c(),
    task_shifting_pharm = 0
  )

  # Ensure JSON-safe output.
  res_json <- lapply(res, function(x) {
    if (is.matrix(x) || is.data.frame(x)) {
      as.data.frame(x)
    } else {
      x
    }
  })

  output_json_path <- "optimizer_results.json"

  jsonlite::write_json(res_json, path = output_json_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
}
