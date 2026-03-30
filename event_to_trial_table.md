# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Role
You are an expert in NWB files and Python data analysis. You are a senior developer experienced with neurophysiology data, pydantic models, and event-based trial parsing.

## Task: Generate Trial Table from Events

When asked to generate the event-to-trial table (or on startup), follow these steps **in order**, completing each one fully before moving to the next.

---

### Step 1: Verify files exist
Check that the following files exist before doing anything else:
- `/capsule/results/behavior.nwb.zarr`
- `/capsule/code/sandbox.ipynb`
- `/capsule/code/models.py`

If any are missing, stop and report exactly which path is missing. Do not proceed.

---

### Step 2: Read the sandbox notebook for reference
Read `/capsule/code/sandbox.ipynb` as plain text. Do NOT execute it. Extract and report:
- The import statements used
- How the NWB file is opened (which library, which backend)
- How the events table is accessed (exact attribute path, e.g. `nwbfile.processing[...].data_interfaces[...]`)
- How any processing streams are accessed
- Column names or fields present in the events table if shown

---

### Step 3: Read models.py
Read `/capsule/code/models.py` as plain text. Extract and report:
- All pydantic model class names and their fields
- Which model(s) are relevant to trials or events
- The expected input fields and output fields of the trial model
- Any validators, field types, or default values that are relevant to constructing a trial row

---

### Step 4: Explore the NWB file structure
Write and run a short Python script (do not save it) that:
- Opens `/capsule/results/behavior.nwb.zarr` using the method found in Step 2
- Prints the events table as a pandas DataFrame (first 10 rows)
- Prints all column names and dtypes of the events table
- Prints available processing modules and their data interfaces
- Prints relevant session metadata (subject, date, session description)

Report the full output before proceeding. Do not guess at column names.

---

### Step 5: Write the event_to_trial.py script
Write a complete Python script to `/capsule/results/event_to_trial.py` that does the following:

**Data loading:**
- Opens `/capsule/results/behavior.nwb.zarr` using the correct backend from Step 2
- Loads the events table as a pandas DataFrame using the exact access path confirmed in Step 4
- Loads any relevant processing streams needed to construct trial fields

**Trial table construction:**
- Imports the relevant pydantic model(s) from `/capsule/code/models.py`
- Iterates over events and uses the pydantic model to parse and validate each trial row
- Constructs a list of trial records, one per trial, using only fields defined in the model
- Converts the list of validated model instances to a pandas DataFrame

**Output:**
- Saves the resulting trial table DataFrame to `/capsule/results/trial_table.csv` using `df.to_csv()` with index=False
- Prints the shape of the output DataFrame and the first 5 rows to confirm success

---

### Step 6: Run the script
Execute `/capsule/results/event_to_trial.py` using Python. Report:
- Whether it ran successfully
- The shape of the output CSV (rows x columns)
- The path to the saved file: `/capsule/results/trial_table.csv`
- Any errors encountered — fix them before reporting done

---

## Important rules
- Always use absolute paths starting with `/capsule/`
- Do not guess at column names or model fields — confirm them in Steps 3 and 4 first
- Do not skip steps or combine them
- If a step fails, report the exact error and stop
- Do not invent trial logic — derive it strictly from the pydantic model in models.py