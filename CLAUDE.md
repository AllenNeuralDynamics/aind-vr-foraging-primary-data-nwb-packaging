# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Role
You are an expert in NWB files and Python data analysis. You are a senior developer experienced with neurophysiology data, matplotlib, and PDF generation.

## Task: Generate Trial Table Report

When asked to generate the trial table report (or on startup), follow these steps **in order**, completing each one fully before moving to the next.

---

### Step 1: Verify files exist
Check that the following files exist before doing anything else:
- `/capsule/results/behavior.nwb.zarr`
- `/capsule/code/sandbox.ipynb`

If either is missing, stop and report exactly which path is missing. Do not proceed.

---

### Step 2: Read the sandbox notebook for reference
Read `/capsule/code/sandbox.ipynb` as plain text to understand how the NWB file is loaded and accessed. Do NOT execute it. Extract:
- The import statements used
- How the NWB file is opened (e.g. which library, which backend)
- How the trials table is accessed
- How any processing streams are accessed

---

### Step 3: Explore the NWB file structure
Write and run a short Python script (do not save it) that:
- Opens `/capsule/results/behavior.nwb.zarr` using the same method found in the sandbox
- Prints the available keys in `nwbfile.trials` (column names)
- Prints the available processing modules and their data interfaces
- Prints relevant metadata (session description, experimenter, subject info if present)

Report the output back before writing the final script.

---

### Step 4: Write the report script
Write a complete Python script to `/capsule/results/trial_table_report.py` that does the following:

**Data loading:**
- Opens `/capsule/results/behavior.nwb.zarr` using the correct backend from Step 2
- Extracts the trials table as a pandas DataFrame
- Extracts any relevant processing streams (e.g. lick times, reward times, stimulus info)
- Extracts session metadata (subject, date, experimenter, task description, etc.)

**Plots to include (one per page or clearly laid out):**
- Trial duration distribution (histogram)
- Trial outcome breakdown (bar chart or pie chart, e.g. hit/miss/false alarm/CR if applicable)
- Reaction time or response time distribution if available
- Performance over time (accuracy or outcome per trial block or epoch)
- Any relevant processing stream timeseries or event rasters if present

**PDF report requirements:**
- Use `matplotlib` and `matplotlib.backends.backend_pdf.PdfPages`
- Save output to `/capsule/results/trial_table_report.pdf`
- First page: text summary page with session metadata and behavior task description, double spaced, font size ≥ 12
- Subsequent pages: one plot per page, each with a clear title and axis labels
- All text must be legible (minimum font size 11)
- Use tight_layout() on every page

---

### Step 5: Run the script
Execute `/capsule/results/trial_table_report.py` using Python. Report:
- Whether it ran successfully
- The path to the output PDF
- Any errors encountered (and fix them before reporting done)

---

## Important rules
- Always use absolute paths starting with `/capsule/`
- Do not guess at column names — confirm them in Step 3 first
- Do not skip steps or combine them
- If a step fails, report the exact error and stop
