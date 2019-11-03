<img src="images/hospital_revolving_door.jpg" width="400" />

# Predicting-Hospital-Readmission-using-NLP
In this project, I build a machine learning model to predict 30-day unplanned hospital re-admission using clinical notes.

# View the full notebook best using nbviewer
I find that nbviewer does a fantastic job of rendering long jupyter notebooks very well (better than Github). For the best experience [view my Notebook here](https://nbviewer.jupyter.org/github/nwams/Predicting-Hospital-Readmission-using-NLP/blob/master/Predicting%20Hospital%20Readmission%20using%20NLP.ipynb).

# Getting Started
To run the Jupyter Notebook, you will need to get access to the MIMIC-III database, a freely available hospital database.
Due to the sensitive nature of medical data, I cannot include the data openly in this repository. If you'd like to get access to the data for this project, you will need to request access at this link: https://mimic.physionet.org/gettingstarted/access/

You will need to download the `ADMISSIONS` and `NOTEEVENTS` tables into this project's folder.

# Blog
I wrote a full blog post for this project as well: https://medium.com/nwamaka-imasogie/predicting-hospital-readmission-using-nlp-5f0fe6f1a705


## Introduction

Now that doctor’s notes are stored in electronic health records, natural language processing can be used for predictive modeling to improve the quality of healthcare. In this project, I build a machine learning model to predict **30-day unplanned** hospital re-admission using clinical notes.

Andrew Long’s [project](https://towardsdatascience.com/introduction-to-clinical-natural-language-processing-predicting-hospital-readmission-with-1736d52bc709) served as the basis for my project, however, in my work I implement several changes that prove to be very valuable and in part 2 of my work I will take it a step further by applying Deep Learning to see if that will improve results. Long’s project was originally inspired by this ArXiv [paper](https://arxiv.org/pdf/1801.07860.pdf) “Scalable and accurate deep learning for electronic health records” by Rajkomar et al.

### Part 1

Both myself and Andrew Long used conventional machine learning models to predict unplanned, 30-day hospital readmissions. My approach **outperformed** the results from Andrew Long’s [project](https://towardsdatascience.com/introduction-to-clinical-natural-language-processing-predicting-hospital-readmission-with-1736d52bc709) by **13% (AUC)**.

As a quick recap, here’s a list of additional things that I did differently:

* Concatenate all the notes (instead of using the last discharge summary only)

* Removed all English stopwords via NLTK

* Performed lemmatization

* Counted readmission only once

### Part 2

In part 2 of this project I will apply a Deep Learning transformer model to see if that will further improve my outcome!

## Intended Audience

This projects is intended for people who are interested in Machine Learning for Healthcare.

## Model Definition

In this project, I build a machine learning model to predict 30-day unplanned hospital readmission using discharge summaries.

**Definitions**:

1. A hospitalization is considered a “re-admission” if its admission date was within 30 days after discharge of a hospitalization.

1. A readmission can only be counted once.

## About the Dataset

I will utilize the MIMIC-III (Medical Information Mart for Intensive Care III), an amazing free hospital database.

The database includes information such as demographics, vital sign measurements made at the bedside (~1 data point per hour), laboratory test results, procedures, medications, caregiver notes, imaging reports, and mortality (including post-hospital discharge).

MIMIC-III is an open-access relational database containing tables of data relating to patients who stayed within the intensive care units (ICU) at Beth Israel Deaconess Medical Center. This public database of Electronic Health Records contains data points on about 41,000 patients from intensive care units between 2001 and 2012, including notes on close to 53,000 admissions.

Due to the sensitive nature of medical data, I cannot include the data openly in this repository. If you’d like to get access to the data for this project, you will need to request access at this link ([https://mimic.physionet.org/gettingstarted/access/)](https://mimic.physionet.org/gettingstarted/access/)).

In this project, I will make use of the following MIMIC tables:

* [**ADMISSIONS](https://mimic.physionet.org/mimictables/admissions/)** — a table containing admission and discharge dates. It has a unique identifier HADM_ID for each admission. HADM_ID refers to a unique admission to the hospital.

* [**NOTEEVENTS](https://mimic.physionet.org/mimictables/noteevents/)** — discharge summaries, which condense information about a patient’s stay into a single document (linked by the HADM_ID). There are a total of 2,083,180 rows in this table.

## Step 1. Prepare the Data

I will follow the steps below to prep the data from the ADMISSIONS and NOTEEVENTS MIMIC tables for my machine learning project.

![](https://cdn-images-1.medium.com/max/2904/1*_LUr3IWPmjbn6ba_VRreXA.png)

### Load ADMISSIONS Table

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    df_adm = pd.read_csv('ADMISSIONS.csv')

### Explore the data

It’s important to always spend time exploring the data.

    df_adm.head()
    df_adm.columns

The main columns of interest in this table are:

* SUBJECT_ID — unique identifier for each subject

* HADM_ID — unique identifier for each hospitalization

* ADMITTIME — admission date with format YYYY-MM-DD hh:mm:ss

* DISCHTIME — discharge date with same format

* DEATHTIME — death time (if it exists) with same format

* ADMISSION_TYPE — includes ELECTIVE, EMERGENCY, NEWBORN, URGENT

Let’s look at the data type of each column

    df_adm.dtypes

Let’s see what’s in the ADMISSION_TYPE column

    df_adm.groupby(['ADMISSION_TYPE']).size()

### Remove NEWBORN admissions

According to the MIMIC [site](https://mimic.physionet.org/mimictables/admissions/) “Newborn indicates that the HADM_ID pertains to the patient’s birth.”

I will remove all NEWBORN admission types because in this project I’m not interested in studying births — my primary interest is EMERGENCY and URGENT admissions.

    df_adm = df_adm.loc[df_adm.ADMISSION_TYPE != 'NEWBORN']

    # Ensure rows with NEWBORN ADMISSION_TYPE are removed
    df_adm.groupby(['ADMISSION_TYPE']).size()

### Remove Deaths

I will remove all admissions that have a DEATHTIME because this in project I’m studying re-admissions, not mortality. And a patient who died cannot be re-admitted.

    # Before removing deaths, first store the hadm_ids for dead patients. It will be used later to also remove deaths from the notes table
    hadm_rows_death = df_adm.loc[df_adm.DEATHTIME.notnull()]
    print("Number of death admissions:", len(hadm_rows_death))

    # Store HADM_ID for dead patients in a list
    hadm_death_list = hadm_rows_death["HADM_ID"].tolist()
    print("Length of the HADM_ID list:", len(hadm_death_list))

Remove admissions of patients who died.

    df_adm = df_adm.loc[df_adm.DEATHTIME.isnull()]

Ensure rows with DEATHTIME are removed

    print('Total rows in admissions dataframe:', len(df_adm))
    print('Non-death admissions:', df_adm.DEATHTIME.isnull().sum())

The two numbers match, which is great because that means that all death admissions were removed successfully.

## Convert strings to dates

According to the MIMIC [website](https://alpha.physionet.org/content/mimiciii/1.4/):
> “…dates were shifted into the future by a random offset for each individual patient in a consistent manner to preserve intervals, resulting in stays which occur sometime between the years 2100 and 2200. Time of day, day of the week, and approximate seasonality were conserved during date shifting. Dates of birth for patients aged over 89 were shifted to obscure their true age and comply with HIPAA regulations: these patients appear in the database with ages of over 300 years.”

When converting dates, it is safer to use a datetime format. Setting the errors = 'coerce' flag allows for missing dates but it sets it to NaT (not a datetime) when the string doesn't match the format. For references on formats see [http://strftime.org/](http://strftime.org/).

    # Convert to dates
    df_adm.ADMITTIME = pd.to_datetime(df_adm.ADMITTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df_adm.DISCHTIME = pd.to_datetime(df_adm.DISCHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df_adm.DEATHTIME = pd.to_datetime(df_adm.DEATHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')

    # Check to see if there are any missing dates
    print('Number of missing admissions dates:', df_adm.ADMITTIME.isnull().sum())
    print('Number of missing discharge dates:', df_adm.DISCHTIME.isnull().sum())

Let’s explore the data types again to ensure they are now datetime, and not object as before.

    print(df_adm.ADMITTIME.dtypes)
    print(df_adm.DISCHTIME.dtypes)
    print(df_adm.DEATHTIME.dtypes)

## Get the next Unplanned admission date for each patient (if it exists)

I need to get the next admission date, if it exists. First I’ll verify that the dates are in order. Then I’ll use the shift() function to get the next admission date.

    # sort by subject_ID and admission date
    df_adm = df_adm.sort_values(['SUBJECT_ID','ADMITTIME'])

    # When we reset the index, the old index is added as a column, and a new sequential index is used. Use the 'drop' parameter to avoid the old index being added as a column
    df_adm = df_adm.reset_index(drop = True)

The dataframe *could* look like this now for a single patient:

![Source: [https://towardsdatascience.com/introduction-to-clinical-natural-language-processing-predicting-hospital-readmission-with-1736d52bc709#5759](https://towardsdatascience.com/introduction-to-clinical-natural-language-processing-predicting-hospital-readmission-with-1736d52bc709#5759)](https://cdn-images-1.medium.com/max/2000/0*KRY5gCYDPxgFH9PJ.png)*Source: [https://towardsdatascience.com/introduction-to-clinical-natural-language-processing-predicting-hospital-readmission-with-1736d52bc709#5759](https://towardsdatascience.com/introduction-to-clinical-natural-language-processing-predicting-hospital-readmission-with-1736d52bc709#5759)*

You can use the groupby and shift operator to get the next admission (if it exists) for each SUBJECT_ID.

    # Create a column and put the 'next admission date' for each subject using groupby. You have to use groupby otherwise the dates will be from different subjects
    df_adm['NEXT_ADMITTIME'] = df_adm.groupby('SUBJECT_ID').ADMITTIME.shift(-1)

    # Same as above. Create a column that holds the 'next admission type' for each subject
    df_adm['NEXT_ADMISSION_TYPE'] = df_adm.groupby('SUBJECT_ID').ADMISSION_TYPE.shift(-1)

![](https://cdn-images-1.medium.com/max/2000/0*ApiBboFBdrFvYaaT.png)

Note that the last admission doesn’t have a next admission.

Since I want to predict **unplanned** re-admissions I will drop (filter out) any future admissions that are **ELECTIVE** so that only EMERGENCY re-admissions are measured.

    # For rows with 'elective' admissions, replace it with NaT and NaN
    rows = df_adm.NEXT_ADMISSION_TYPE == 'ELECTIVE'
    df_adm.loc[rows,'NEXT_ADMITTIME'] = pd.NaT
    df_adm.loc[rows,'NEXT_ADMISSION_TYPE'] = np.NaN

![](https://cdn-images-1.medium.com/max/2000/0*L7fsbyFvYkgZuHV8.png)

Backfill in the values that I removed. So copy the ADMITTIME from the last emergency and paste it in the NEXT_ADMITTIME for the **previous** emergency. So I am effectively ignoring/skipping the ELECTIVE admission row completely. Doing this will allow me to calculate the days until the next admission.

    # Sort by subject_id and admission date
    # It's safer to sort right before the fill incase something I did above changed the order
    df_adm = df_adm.sort_values(['SUBJECT_ID','ADMITTIME'])

    # Back fill. This will take a little while.
    df_adm[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']] = df_adm.groupby(['SUBJECT_ID'])[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']].fillna(method = 'bfill')

![](https://cdn-images-1.medium.com/max/2000/0*L9kJKjckaiRsEYXI.png)

## Calculate days until next admission

Now let’s calculate the **days** between discharge and the next emergency visit.

    df_adm['DAYS_TIL_NEXT_ADMIT'] = (df_adm.NEXT_ADMITTIME - df_adm.DISCHTIME).dt.total_seconds()/(24*60*60)

Now I’ll count the total number of re-admissions in the dataset. This includes EMERGENCY and URGENT re-admissions because according to the MIMIC [website](https://mimic.physionet.org/mimictables/admissions/) *“Emergency/urgent indicate unplanned medical care, and are often collapsed into a single category in studies.”*

In the dataset there are 45,321 hospitalizations with 9,705 re-admissions. For those with re-admissions I can plot the histogram of days between admissions.

    plt.hist(df_adm.loc[~df_adm.DAYS_TIL_NEXT_ADMIT.isnull(),'DAYS_TIL_NEXT_ADMIT'], bins =range(0,365,30))
    plt.xlim([0,365])
    plt.xlabel('Days between admissions')
    plt.ylabel('Counts')
    plt.xticks(np.arange(0, 365, 30))
    plt.title('Histogram of 30-day unplanned re-admissions over 365 days')
    plt.show()

![](https://cdn-images-1.medium.com/max/2000/1*SuFW2aDc_AF9sXUayhn4Nw.png)

## Load NOTEEVENTS Table

Now I’m ready to work with the NOTEEVENTS table.

    df_notes = pd.read_csv("NOTEEVENTS.csv")

### I’ll spend some time exploring the data

There are 2,083,180 notes. The number of notes is much higher than the number of hospitalizations (45,321) because there can be multiple notes per hospitalization.

The main columns of interest from the [NOTEEVENTS](https://mimic.physionet.org/mimictables/noteevents/) table are:

* SUBJECT_ID

* HADM_ID

* CHARTDATE — records the date at which the note was charted. CHARTDATE will always have a time value of 00:00:00.

* CATEGORY — defines the type of note recorded.

* TEXT — contains the note text.

There are 231,836 null HADM_ID values. This means that over 11% of the notes are missing unique hospitalization identifiers (HADM_IDs). This seems like it could be problematic so I'll investigate what might be causing this later on.

The good news is all records have a CHARTDATE and all records have TEXT values.

Let’s see what type of information is in the CATEGORY column.

    df_notes['CATEGORY'].value_counts()

The CATEGORY column contains Nursing/other, Radiology, Nursing, ECG, Physician, Discharge summary, Echo, Respiratory, Nutrition, General, Rehab Services, Social Work, Case Management, Pharmacy and Consult.

Let’s look at the contents of the first note from the TEXT column.

    df_notes.TEXT.iloc[0]

Due to data privacy agreements I can’t show the contents of the individual notes, but I will just describe them. In the notes, the dates and any *Protected Health Information* like name, doctor and location have been converted for confidentiality. There are also newline characters \n, numbers and punctuation.

## Investigate Why HADM_ID’s are missing

Before going any further, I need to figure out why approx. 11% of the notes are missing.

I found an **important** discovery on the MIMIC [site](https://mimic.physionet.org/mimictables/noteevents/):

* If a patient is an **outpatient**, there will not be an HADM_ID associated with the note — An outpatient is a patient who receives medical treatment without being admitted to a hospital

* If the patient is an **inpatient**, but was not admitted to the ICU for that particular hospital admission, then there will not be an HADM_ID associated with the note — An inpatient is a patient who stays in a hospital while under treatment.

This explains why some HADM_IDs are missing. Let’s move on.

## Remove notes for death admissions

Before I concatenate notes for each patient, I need to remove the death admission notes as well. So that notes for dead patients don’t influence my model later. Remember that earlier I removed deaths from the df_adm dataframe so now I'll something similar for df_notes.

    df_notes = df_notes[~df_notes['HADM_ID'].isin(hadm_death_list)]

There are 1,841,968 notes remaining after deaths were removed.

## Concatenate Notes for Each Patient

Since there are multiple notes per hospitalization, I will decide to concatenate **all** of notes that belong to a patient. An alternative approach would be to use just the discharge summary, however I wanted to give the Deep Learning algorithm a full taste of the data. This will give the Neural Network the opportunity to extract insights about which features are relevant.

When concatenating the notes, I want to maintain chronological order. To determine the order I’ll use the CHARTDATE column along with CHARTTIME, if it's available, because some CHARTTIME entries have missing (null) values according to the MIMIC [site](https://mimic.physionet.org/mimictables/noteevents/).

Convert the dates to datetime format.

    # Convert CHARTDATE from string to a datetime format
    df_notes.CHARTDATE = pd.to_datetime(df_notes.CHARTDATE, format = '%Y-%m-%d', errors = 'coerce')

    # Convert CHARTTIME to datetime format
    df_notes.CHARTTIME = pd.to_datetime(df_notes.CHARTTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')

Now that the dates were converted successfully, now I can sort each patient’s notes in order of the date that it was entered. If the dates happen to be the same I will order it by whichever comes first in the table.

    # Sort by subject_ID, CHARTDATE then CHARTTIME
    df_notes = df_notes.sort_values(['SUBJECT_ID','CHARTDATE', 'CHARTTIME'])
    df_notes = df_notes.reset_index(drop = True)

Create a new dataframe that contains the columns: SUBJECT_ID, and all concatenated notes for each patient. I am basically squashing all notes down to one SUBJECT_ID.

    # Copy over two columns to new dataframe
    df_subj_concat_notes = df_notes[['SUBJECT_ID', 'TEXT']].copy()

Concatenate notes that belong to the same SUBJECT_ID and compress the SUBJECT_ID. Special thanks to this great Stack Overflow [answer](https://stackoverflow.com/questions/53781634/aggregation-in-pandas) on performing string aggregation.

    df_subj_concat_notes = df_subj_concat_notes.groupby('SUBJECT_ID')['TEXT'].agg(' '.join).reset_index()

    # Rename the column in new dataframe to TEXT_CONCAT
    df_subj_concat_notes.rename(columns={"TEXT":"TEXT_CONCAT"}, inplace=True)

Since the next step is to merge the notes with the admissions table, let’s double check that every SUBJECT_ID is unique (no duplicates). I can check this with an assert statement.

    assert df_subj_concat_notes.duplicated(['SUBJECT_ID']).sum() == 0, 'Dulpicate SUBJECT_IDs exist'

There are no errors. And there are no missing TEXT fields. Let’s keep on pushing.

**Remember**: Earlier I removed Newborns from the admission’s table. I need to also remove Newborn remnants from this table as well to ensure they are not considered. But if there was any trace of newborns left, it would automatically take care of itself when I left-merge the two tables together. The same is true for deaths, the merge should automatically handle this as well.

## Merge Datasets

Now I’m ready to merge the admissions and concatenated notes tables. I use a left merge. There are a lot of cases where you get multiple rows after a merge (although we dealt with it above), it’s a good idea to be extra careful and add assert statements after a merge.

    df_adm_notes = pd.merge(df_adm[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','DAYS_TIL_NEXT_ADMIT','NEXT_ADMITTIME','ADMISSION_TYPE','DEATHTIME']],
                            df_subj_concat_notes, 
                            on = ['SUBJECT_ID'],
                            how = 'left')
    assert len(df_adm) == len(df_adm_notes), 'Number of rows increased'

Looks like the merge went as expected. You’ll notice that in the df_adm_notes table that even though there cases with duplicate SUBJECT_IDs each duplicate SUBJECT_ID still contains identical notes. This is almost how I want it but not quite.

Instead of identical notes for duplicate SUBJECT_IDs I'd like to only keep one note per SUBJECT_ID (so that when I apply Bag-of-Words later the words counts will be correct/not duplicated).

49 admissions (0.11%) don’t have notes. I could do more digging but since this is less than 1% let’s move forward.

## Make Output Label

For this problem, we are going to classify if a patient will be admitted in the next 30 days. Therefore, we need to create a variable with the output label (1=readmitted, 0=not readmitted).

Now I’ll create a column in the dataframe called OUTPUT_LABEL that holds the predictions:

* 1 for re-admitted

* 0 for not re-admitted.

**Remember** we want to predict if the patient was re-admitted **within 30 days**. A hospitalization is considered a “re-admission” if its admission date was within 30 days after discharge of a hospitalization.

    # Create new column of 1's or 0's based on DAYS_TIL_NEXT_ADMIT
    df_adm_notes['OUTPUT_LABEL'] = (df_adm_notes.DAYS_TIL_NEXT_ADMIT < 30).astype('int')

I’ll take a quick count of positive and negative results.

    print('Number of positive samples:', (df_adm_notes.OUTPUT_LABEL == 1).sum())
    print('Number of negative samples:', (df_adm_notes.OUTPUT_LABEL == 0).sum())
    print('Total samples:', len(df_adm_notes))

There are 2,549 positive samples and 42,772 negative samples which means this is an imbalanced dataset — which is very common in healthcare analytics projects.

Now, I will squash SUBJECT_IDs so that there is only one SUBJECT_ID per patient (no multiples). Remember in the problem definition at the beginning (in-line with this source [paper](https://arxiv.org/pdf/1801.07860.pdf) that inspired this project), readmissions can only be counted once. So as long as a patient has had one readmission that falls within a 30 day window, they will receive an output label of 1, and all other patients who either were readmitted but not within 30 days or who were never readmitted will receive an output label of 0.

The only columns that are important for me to carry on moving forward now are:

* SUBJECT_ID

* TEXT_CONCAT

* OUTPUT_LABEL

    # Take only the 3 essential columns
    df_adm_notes_squashed = df_adm_notes[['SUBJECT_ID', 'TEXT_CONCAT', 'OUTPUT_LABEL']]

Create a new dataframe that officially squashes (compresses) the SUBJECT_ID column. Then sum the output labels. Notice that during the squash, the TEXT_CONCAT notes are not taken over.

I will merge the newly created df_subj_labels_squashed with df_adm_notes_squashed later to fix that.

    df_subj_labels_squashed = df_adm_notes_squashed.groupby('SUBJECT_ID')[['OUTPUT_LABEL']].sum().reset_index()

Rename the column in the new dataframe to OUTPUT_LABELS_SUMMED.

    df_subj_labels_squashed.rename(columns={"OUTPUT_LABEL":"OUTPUT_LABELS_SUMMED"}, inplace=True)

Set 1 to OUTPUT_LABEL if the OUTPUT_LABELS_SUMMED are greater than or equal to 1. This essentially means that several readmissions per patient are counted only once.

    df_subj_labels_squashed['OUTPUT_LABEL'] = (df_subj_labels_squashed['OUTPUT_LABELS_SUMMED'] >= 1).astype(int)

Drop the OUTPUT_LABELS_SUMMED column as it's no longer needed.

    df_subj_labels_squashed.drop(columns=['OUTPUT_LABELS_SUMMED'], inplace=True)

Before merging drop the OUTPUT_LABEL from the original df_adm_notes_squashed table. It's no longer needed because the OUTPUT_LABEL on the df_subj_labels_squashed is the correct one we'll use moving forward.

    df_adm_notes_squashed.drop(columns=['OUTPUT_LABEL'], inplace=True)

Prepping for merge: Drop duplicates in df_adm_notes_squashed.

    df_adm_notes_squashed.drop_duplicates(subset='SUBJECT_ID', keep='first', inplace=True)

Check that the two dataframes are of equal length before merging.

    print('Length of df_adm_notes_squashed:', len(df_adm_notes_squashed))
    print('Length of df_subj_labels_squashed:', len(df_subj_labels_squashed))

Merge the two tables so we can get our notes back alongside the output_label.

    df_adm_notes_merged = pd.merge(df_subj_labels_squashed[['SUBJECT_ID','OUTPUT_LABEL']],
                            df_adm_notes_squashed, 
                            on = ['SUBJECT_ID'],
                            how = 'left')
    assert len(df_subj_labels_squashed) == len(df_adm_notes_merged), 'Number of rows increased'

## Make Training / Validation / Test sets

Split the data into training, validation and test sets.

1. Training set — used to train the model.

1. Validation set — data that the model hasn’t seen, but it is used to optimize/tune the model.

1. Test set — data that both the model and tuning process have never seen. It is the true test of generalizability.

The validation and test set should be as close to the production data as possible. We don’t want to make decisions on validation data that is not from same type of data as the test set.

I’ll set the random_state = 42 for reproduciblity so that I can benchmark my results against Andrew Long's [results](https://towardsdatascience.com/introduction-to-clinical-natural-language-processing-predicting-hospital-readmission-with-1736d52bc709).

Shuffle the samples.

    df_adm_notes_merged = df_adm_notes_merged.sample(n=len(df_adm_notes_merged), random_state=42)
    df_adm_notes_merged = df_adm_notes_merged.reset_index(drop=True)

The SUBJECT_IDs are no longer in ascending order, so this indicates the shuffle went as planned.

Randomly split patients into training sets (80%), validation sets (10%), and test (10%) sets.

    df_valid_and_test = df_adm_notes_merged.sample(frac=0.20, random_state=42)

    df_test = df_valid_and_test.sample(frac=0.5, random_state=42)
    df_valid = df_valid_and_test.drop(df_test.index)

    df_train = df_adm_notes_merged.drop(df_valid_and_test.index)

    assert len(df_adm_notes_merged) == (len(df_test)+len(df_valid)+len(df_train)),"Split wasn't done mathetmatically correct."

### Prevalence

In the medical world prevalence is defined as the proportion of individuals in a population having a disease or characteristic. Prevalence is a statistical concept referring to the number of cases of a disease that are present in a particular population at a given time.

    print("Training set prevalence (n = {:d}):".format(len(df_train)), "{:.2f}%".format((df_train.OUTPUT_LABEL.sum()/len(df_train))*100))

    print("Validation set prevalence (n = {:d}):".format(len(df_valid)), "{:.2f}%".format((df_valid.OUTPUT_LABEL.sum()/len(df_valid))*100))

    print("Test set prevalence (n = {:d}):".format(len(df_test)), "{:.2f}%".format((df_test.OUTPUT_LABEL.sum()/len(df_test))*100))

    print("All samples (n = {:d})".format(len(df_adm_notes_merged)))

The prevalence’s are low.

In Machine Learning if the prevalence is too low we will need to balance the training data to prevent our model from always predicting negative (not re-admitted). To balance the data, we have a few options:

1. Sub-sample the negatives

1. Over-sample the positives

1. Create synthetic data (e.g. SMOTE)

In line with the author’s choice (for benchmarking purposes) I will **sub-sample negatives for the training set**.

    # Split the training data into positive and negative outputs
    pos_rows = df_train.OUTPUT_LABEL == 1
    df_train_pos = df_train.loc[pos_rows]
    df_train_neg = df_train.loc[~pos_rows]

    # Merge the balanced data
    df_train = pd.concat([df_train_pos, df_train_neg.sample(n = len(df_train_pos), random_state=42)], axis = 0)

    # Shuffle the order of training samples
    df_train = df_train.sample(n = len(df_train), random_state = 42).reset_index(drop=True)

    print("Training set prevalence (n = {:d}):".format(len(df_train)), "{:.2f}%".format((df_train.OUTPUT_LABEL.sum()/len(df_train))*100))

Now the training set prevalence is 50% so the new training set is balanced (although I reduced the size significantly in order to do so).

## Step 2: Preprocessing Pipeline

![Bag of Words (BOW) -> Count Vectorizer](https://cdn-images-1.medium.com/max/2104/1*Yz8-i390Qf6qo6QEuPOVXQ.png)*Bag of Words (BOW) -> Count Vectorizer*

First I’ll try the Bag-of-Words (BOW) approach. Bag of words is just a group of text where the order doesn’t matter. BOW just counts the occurrence of words. It’s a simple and powerful way to represent text data. Ultimately, I’ll feed this into the machine learning model.

1. Lowercase text

1. Remove punctuation

1. Remove numbers and words that contain numbers

1. Remove newline characters \n and carriage returns \r

1. Tokenize the text

1. Remove stop words

1. Lemmatize — which is reducing words like “driving, drive, drives” down to it’s base word drive (using Part-of-Speech tagging). It reduces the feature space making models more performant.

**Side note:** The preprocessing pipeline that I’ll do for conventional machine learning models now will differ from the preprocessing required for Deep Learning in part 2.

### Clean and Tokenize

    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import string
    import re

    from nltk import pos_tag
    from nltk.stem import WordNetLemmatizer # lemmatizes word based on its parts of speech

    print('Punctuation:', string.punctuation)
    print('NLTK English Stop Words:', '\n', stopwords.words('english'))

### Lemmatize

**convert_tag**

The POS tags used by the part-of-speech tagger are not the same as the POS codes used by WordNet, so I need a small mapping function convert_tag to convert POS tagger tags to WordNet POS codes. To convert Treebank tags to WordNet tags the mapping is as follows:
> wn.VERB = 'v'
wn.ADV = 'r'
wn.NOUN = 'n'
wn.ADJ = 'a'
wn.ADJ_SAT = 's'

But we can ignore ‘s’ because the WordNetLemmatizer in NLTK does not differentiate satellite adjectives from normal adjectives. See the WordNet [docs](http://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html). The other parts of speech will be tagged as nouns. See this [post](https://stackoverflow.com/questions/51634328/wordnetlemmatizer-different-handling-of-wn-adj-and-wn-adj-sat) if you’re interested in details.

    def convert_tag(treebank_tag):
        '''Convert Treebank tags to WordNet tags'''
        if treebank_tag.startswith('J'):
            return 'a'
        elif treebank_tag.startswith('V'):
            return 'v'
        elif treebank_tag.startswith('N'):
            return 'n'
        elif treebank_tag.startswith('R'):
            return 'r'
        else:
            return 'n' # if no match, default to noun

By default, the WordNetLemmatizer.lemmatize() function will assume that the word is a Noun if there's no explicit POS tag in the input. To resolve the problem, always POS-tag your data before lemmatizing.

    def lemmatizer(tokens):
        '''
        Performs lemmatization.
        Params:
            tokens (list of strings): cleaned tokens with stopwords removed
        Returns:
            lemma_words (list of strings): lemmatized words
        '''  
        # POS-tag your data before lemmatizing
        tagged_words = pos_tag(tokens) # outputs list of tuples [('recent', 'JJ'),...]
        
        
        # Lemmatize using WordNet's built-in morphy function. Returns the input word unchanged if it cannot be found in WordNet.
        wnl = WordNetLemmatizer()
        
        lemma_words = []
        
        # Lemmatize list of tuples, output a list of strings
        for tupl in tagged_words:
            lemma_words.append(wnl.lemmatize(tupl[0], convert_tag(tupl[1])))
        
        return lemma_words

    def preprocess_and_tokenize(text):
        '''
        Clean the data.
        Params:
            text (string): full original, uncleaned text
        Returns:
            lemmatized_tokens (list of strings): cleaned words
        '''
        # Make text lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        
        # Remove numbers and words that contain numbers
        text = re.sub('\w*\d\w*', '', text)
        
        # Remove newline chars and carriage returns
        text = re.sub('\n', '', text)
        text = re.sub('\r', '', text)
        
        # Tokenize
        word_tokens = word_tokenize(text) 
        
        # Remove stop words
        tokens = [word for word in word_tokens if word not in stopwords.words('english')]

    # Call lemmatizer function above to perform lemmatization
        lemmatized_tokens = lemmatizer(tokens)
        
        return lemmatized_tokens

## Count Vectorizer

Now that I have functions that convert the notes into tokens I’ll use CountVectorizer from scikit-learn to count the tokens for each patient's concatenated notes. **NOTE:** There is also a TfidfVectorizer which takes into account how often words are used across all notes, but for this project, I'll use the same/simpler one that the author used (the author also got similar results with the TfidfVectorizer one too).

### Build a vectorizer on the clinical notes

Now I’ll fit the CountVectorizer on the clinical notes. Remember **only use the training data**.

Set the hyperparameter, max_features, which will use the top N most frequently used words. Later on I'll tune this to see its effect.

If you’re interested in learning more this [blog](https://towardsdatascience.com/natural-language-processing-count-vectorization-with-scikit-learn-e7804269bb5e) explains CountVectorizerization with scikit learn.

    from sklearn.feature_extraction.text import CountVectorizer

    vect = CountVectorizer(max_features=3000, tokenizer=preprocess_and_tokenize)

This could take a while depending on your computer, took several hours.

    # I applied .astype(str) to fix the ValueError: np.nan is an invalid document, expected byte or unicode string.xc

    # create the vectorizer
    vect.fit(df_train.TEXT_CONCAT.values.astype(str))

### Zipf’s Law

Zipf’s law tells you how many frequent words and rare words you are going to have in a collection of text. Here’s a short [video](https://www.youtube.com/watch?v=KvOS2MdKFwE) that explains it well.

Actually create a vector by passing the text into the vectorizer to get back counts.

    neg_doc_matrix = vect.transform(df_train[df_train.OUTPUT_LABEL == 0].TEXT_CONCAT.astype(str))
    pos_doc_matrix = vect.transform(df_train[df_train.OUTPUT_LABEL == 1].TEXT_CONCAT.astype(str))

Sum over the columns.

    neg_tf = np.sum(neg_doc_matrix,axis=0)
    pos_tf = np.sum(pos_doc_matrix,axis=0)

Remove the non-useful one dimension from array. Helps you get rid of useless one dimension arrays e.g. [7,8,9] instead of [[[7,8,9]]].

    neg = np.squeeze(np.asarray(neg_tf))
    pos = np.squeeze(np.asarray(pos_tf))

Now I will transform the notes into **numerical matrices**. I’m still only going to use the training and validation data, not the test set yet.

    # Could take a while
    X_train_tf = vect.transform(df_train.TEXT_CONCAT.values.astype(str))
    X_valid_tf = vect.transform(df_valid.TEXT_CONCAT.values.astype(str))

Get the output labels as separate variables.

    y_train = df_train.OUTPUT_LABEL
    y_valid = df_valid.OUTPUT_LABEL

Now I’m finally done prepping the data for the predictive model.

## Step 3: Build a simple predictive model

Now I will build a simple predictive model that takes the bag-of-words as inputs and predicts if a patient will be readmitted in 30 days (YES = 1, NO = 0).

I will use the **Logistic Regression** model from scikit-learn. Logistic regression is a good baseline model for NLP tasks because it is interpretable and works well with sparse matrices.

I will tune 2 hyperparameters: the C coefficient and penalty:

* C — Coefficient on regularization where smaller values specify stronger regularization.

* Penalty — tells how to measure the regularization.

Regularization is a technique to try to minimize overfitting. I wrote about it in this [section](https://medium.com/nwamaka-imasogie/neural-networks-word-embeddings-8ec8b3845b2e#4358) of one of my blogs.

    from sklearn.linear_model import LogisticRegression

    # Classifier
    clf = LogisticRegression(C = 0.0001, penalty = 'l2', random_state = 42)
    clf.fit(X_train_tf, y_train)

[Calculate the probability](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_proba) of readmission for each sample with the fitted model.

    model = clf
    y_train_preds = model.predict_proba(X_train_tf)[:,1]
    y_valid_preds = model.predict_proba(X_valid_tf)[:,1]

Show the first 10 Training Output Labels and their Probability of Readmission.

    df_training_prob = pd.DataFrame([y_train[:10].values, y_train_preds[:10]]).transpose()
    df_training_prob.columns = ['Actual', 'Probability']
    df_training_prob = df_training_prob.astype({"Actual": int})

## Visualize top words for positive and negative classes.

This code snippet below from [here](https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb).

To validate my model and interpret its predictions, it is important to look at which words it is using to make decisions, to see if there are any patterns which could give insight into additional features to add or remove.

I’ll plot the most important words for both the negative and positive class. Plotting word importance is simple with Bag of Words and Logistic Regression, since we can just extract and rank the coefficients that the model used for its predictions.

    def get_most_important_features(vectorizer, model, n=5):
        index_to_word = {v:k for k,v in vectorizer.vocabulary_.items()}
        
        # loop for each class
        classes ={}
        for class_index in range(model.coef_.shape[0]):
            word_importances = [(el, index_to_word[i]) for i,el in enumerate(model.coef_[class_index])]
            sorted_coeff = sorted(word_importances, key = lambda x : x[0], reverse=True)
            tops = sorted(sorted_coeff[:n], key = lambda x : x[0])
            bottom = sorted_coeff[-n:]
            classes[class_index] = {
                'tops':tops,
                'bottom':bottom
            }
        return classes

    def plot_important_words(top_scores, top_words, bottom_scores, bottom_words, name):
        y_pos = np.arange(len(top_words))
        top_pairs = [(a,b) for a,b in zip(top_words, top_scores)]
        top_pairs = sorted(top_pairs, key=lambda x: x[1])
        
        bottom_pairs = [(a,b) for a,b in zip(bottom_words, bottom_scores)]
        bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)
        
        top_words = [a[0] for a in top_pairs]
        top_scores = [a[1] for a in top_pairs]
        
        bottom_words = [a[0] for a in bottom_pairs]
        bottom_scores = [a[1] for a in bottom_pairs]
        
        fig = plt.figure(figsize=(10, 15))

    plt.subplot(121)
        plt.barh(y_pos,bottom_scores, align='center', alpha=0.5)
        plt.title('Negative', fontsize=20)
        plt.yticks(y_pos, bottom_words, fontsize=14)
        plt.suptitle('Key words', fontsize=16)
        plt.xlabel('Importance', fontsize=20)
        
        plt.subplot(122)
        plt.barh(y_pos,top_scores, align='center', alpha=0.5)
        plt.title('Positive', fontsize=20)
        plt.yticks(y_pos, top_words, fontsize=14)
        plt.suptitle(name, fontsize=16)
        plt.xlabel('Importance', fontsize=20)
        
        plt.subplots_adjust(wspace=0.8)
        plt.show()

    importance = get_most_important_features(vect, clf, 50)

    top_scores = [a[0] for a in importance[0]['tops']]
    top_words = [a[1] for a in importance[0]['tops']]
    bottom_scores = [a[0] for a in importance[0]['bottom']]
    bottom_words = [a[1] for a in importance[0]['bottom']]

    plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words")

![](https://cdn-images-1.medium.com/max/2000/1*E7h7r41dXFxw5IHQViMcuw.png)

Not bad, let’s move forward.

## Step 4: Calculate Performance Metrics

To assess the quality of the model we need to measure how well our model performed. In this project I’ll choose the the AUROC metric (also known as AUC) because:

1. For benchmarking purposes — it is the metric used in the original [Arxiv](https://arxiv.org/pdf/1801.07860.pdf) paper that inspired this project.

1. It balances False Positive Rates and True Positive Rate. See below for more details.

The **ROC curve** shows how the **recall** vs **precision** relationship changes as we vary the threshold for identifying a positive in our model.

An ROC curve plots the true positive rate on the y-axis versus the false positive rate on the x-axis. The **true positive rate (TPR)** is the **recall** and the **false positive rate (FPR)** is the **probability of a false alarm**. Both of these can be calculated from the confusion matrix:

![](https://cdn-images-1.medium.com/max/2000/1*Nu29gRsSO-45gR-J9cDyiw.png)

![[[Source](http://Source)]](https://cdn-images-1.medium.com/max/2404/1*cwcHM66BdXun7UtwT83rBw.png)*[[Source](http://Source)]*

* **Recall** expresses the ability of a model to find all the relevant cases within a dataset.

* **Precision** expresses the ability of a classification model to identify only the relevant data points. In other words, the proportion of the data points our model says was relevant actually were relevant.

![](https://cdn-images-1.medium.com/max/2708/1*SGx9U6VsgLv8riLOCu6flg.png)

**NOTE**: Although accuracy is the most common performance metric, it is not the best metric to use in this project because, for example, if you always predict that people will not be hospitalized you might achieve an extremely high accuracy, but **you will not predict any of the actual hospitalizations**.

After setting a threshold of 0.5 for predicting positive, I get the following metrics:

![](https://cdn-images-1.medium.com/max/2000/1*mFOvtxogSjvuPuu080K2fQ.png)

Notice that there is a significant drop in the precision of the training data versus the validation data. This is because earlier on I balanced the training set by setting the prevalence to 50%, however, the validation remained the original distribution.

A well performing classifier has a *low* classification error but a *high* AUC [[Source](https://www.researchgate.net/profile/David_Tax/publication/221275639_Learning_Curves_for_the_Analysis_of_Multiple_Instance_Classifiers/links/0fcfd5138ac2f3257b000000.pdf#page=8)].

Now I’ll plot the Receiver Operating Characteristic (ROC) curve.

![](https://cdn-images-1.medium.com/max/2000/1*6qtyPjtFOopJIXnFYIEaxA.png)

Notice that I do have some overfitting.

## Evaluating Classifier Performance

When you’ve finally reached the point where you’re thinking of ways to improve the model, we can do it in a data-driven way to avoid spending a lot of time going down the wrong path.

Check out Andrew Ng’s *Deep Learning* Coursera videos where he discusses high-bias vs. high-variance [[Week 1, videos 2 & 3](https://www.coursera.org/learn/deep-neural-network/lecture/ZhclI/bias-variance)]. Additionally, in [[Week 6, videos 1 & 3](https://www.coursera.org/learn/machine-learning/home/week/6)] of Andrew Ng’s *Machine Learning* Course he covers **learning curves**.

A learning curve shows when the continuation of learning has no further effects.

If a learning algorithm is suffering from **high bias (underfitting)**, getting more training data will not (by itself) help much — notice that the **error** remains high. This is shown in Andrew Ng’s [slide](https://www.coursera.org/learn/machine-learning/lecture/Kont7/learning-curves) below:

![[[Source](https://www.coursera.org/learn/machine-learning/lecture/Kont7/learning-curves)]](https://cdn-images-1.medium.com/max/2000/1*pa8JTdAI3KU1uxfIQia-Hg.png)*[[Source](https://www.coursera.org/learn/machine-learning/lecture/Kont7/learning-curves)]*

We can tell that an algorithm is suffering from **high variance (overfitting)** if there’s a “large” gap between the training error curve and the cross-validation error curve. If a learning algorithm is suffering from high variance, getting more training data is likely to help because if you were to extrapolate to the right the two errors start to converge.

![[[Source](https://www.coursera.org/learn/machine-learning/lecture/Kont7/learning-curves)]](https://cdn-images-1.medium.com/max/2000/1*uMNe_ADK4JfQt8s1XAMFfg.png)*[[Source](https://www.coursera.org/learn/machine-learning/lecture/Kont7/learning-curves)]*

## Plot the Learning Curve

![](https://cdn-images-1.medium.com/max/2000/1*2a4AgAYZpwBF5MJirttqnw.png)

As you can see from the learning curve above:

1. There is some overfitting but

1. I am convinced that adding more data will improve the results. This is confirmed by the fact that the cross-validation score trend seems to be increasing as we increase the training examples. It appears that it is trending toward converging w the training score if we were to keep extrapolating. Unfortunately I cannot obtain more data for this project because I’m limited to what the MIMIC-III has made available. So let’s move on.

## Hyperparameter Tuning

In general machine learning is a very iterative process where you can tweak hyperparameters to find the optimal results.

Tuning is vital to enhance your models performance. To optimize my hyperparameters I could either fit a [Grid Search](https://chrisalbon.com/machine_learning/model_selection/hyperparameter_tuning_using_grid_search/) on my training data or a [Random Search](https://towardsdatascience.com/hyperparameter-tuning-c5619e7e6624#76a2) on my training data but I won’t perform them because of the lengthy computing time. previously training my model took on average over 12 hours per task (it’s lengthy likely due to the fact that I concatenated the notes). Unfortunately due to compute restraints it might take several days to perform hyperparameter tuning. So instead, now I will just discuss the two methods as well as their advantages and disadvantages.

### Grid Search

* Grid search is an approach that methodically builds a and evaluates a model for each combination of algorithm parameters that you will have to define in a grid.

* You can use sklearn's [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

* For example with Logistic Regression suppose you define the grid as:

* penalty = [‘l1’, ‘l2’]

* C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

* It will start with the combination of [‘l1’, 0.0001], and it will end with [‘l2’, 1000]. It will go through all intermediate combinations between these two which makes grid search **very computationally expensive**.

### Random Search

* Random search searches the specified subset of hyperparameters randomly instead of exhaustively.

* The major benefit, compared to grid search, is its **decreased computation time**.

* The tradeoff however is that we are not guaranteed to find the optimal combination of hyperparameters because it didn’t exhaustively search all possible combinations.

* You can use sklearn's [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html). An important additional parameter to specify here is n_iter which is the number of combinations to randomly try. Selecting a number that's too low will decrease our chance of finding he best combination, but selecting a number that's too large will increase compute time.

## What comes after tuning?

After finding your optimal hyperparmeters you should plug in the hyperparameter values to your **validation set** and keep the best trained model.

Then finally you would run your model on the test set with those same optimal hyperparameters plugged in.

**Side notes:**
> *If your project requires that you take it a step further and compare different models in an unbiased way, you can finally use the test set for this. See more in this Stack Exchange [answer](https://datascience.stackexchange.com/questions/18339/why-use-both-validation-set-and-test-set).*
> *You cannot use the cross validation set to measure performance of your model accurately, because you will deliberately tune your results to get the best possible metric, over maybe hundreds of variations of your parameters. The cross validation result is therefore likely to be too optimistic. See this Stack Exchange [answer](https://datascience.stackexchange.com/a/18347) as well for more robust explanation.*

## Step 6: Run Model on Test set

    # Could take some hours
    X_test_tf = vect.transform(df_test.TEXT_CONCAT.values.astype(str))

    # Get the output labels as separate variables.
    y_test = df_test.OUTPUT_LABEL

    # Calculate the probability of readmission for each sample with the fitted model
    y_test_preds = model.predict_proba(X_test_tf)[:,1]

    # Print performance metrics
    test_recall = calc_recall(y_test, y_test_preds, thresh)
    test_precision = calc_precision(y_test, y_test_preds, thresh)
    test_prevalence = calc_prevalence(y_test)
    auc_test = roc_auc_score(y_test, y_test_preds)

    print('Train prevalence(n = %d): %.3f'%(len(y_train), train_prevalence))
    print('Valid prevalence(n = %d): %.3f'%(len(y_valid), valid_prevalence))
    print('Test prevalence(n = %d): %.3f'%(len(y_test), test_prevalence))

    print('Train AUC:%.3f'%auc_train)
    print('Valid AUC:%.3f'%auc_valid)
    print('Valid AUC:%.3f'%auc_test)

    print('Train recall:%.3f'%train_recall)
    print('Valid recall:%.3f'%valid_recall)
    print('Test recall:%.3f'%test_recall)

    print('Train precision:%.3f'%train_precision)
    print('Valid precision:%.3f'%valid_precision)
    print('Test precision:%.3f'%test_precision)

    # Quickly throw metrics into a table format. It's easier to eyeball it for a side-by-side comparison.
    df_test_perf_metrics = pd.DataFrame([
        [train_prevalence, valid_prevalence, test_prevalence],
        [train_recall, valid_recall, test_recall], 
        [train_precision, valid_precision, test_precision],
        [auc_train, auc_valid, auc_test]], columns=['Training', 'Validation', 'Test'])

    df_test_perf_metrics.rename(index={0:'Prevalence', 1:'Recall', 2:'Precision', 3:'AUC'}, 
                     inplace=True)

    df_test_perf_metrics.round(3)

This produces the following results.

![](https://cdn-images-1.medium.com/max/2000/1*urnSFwo9vCqN04Na82Dg2Q.png)

Plot the ROC Curve.

![](https://cdn-images-1.medium.com/max/2000/1*ZsfmcfOoOoGgew2ev0Yw1g.png)

## Conclusion — Benchmark the Results

It’s time to compare results. Both me and Andrew Long used conventional machine learning models to predict unplanned, 30-day hospital readmissions. My approach (AUC=0.83) **outperformed** Andrew Long’s [results](https://towardsdatascience.com/introduction-to-clinical-natural-language-processing-predicting-hospital-readmission-with-1736d52bc709) (AUC=0.70) by **13%**.

As a quick recap, here’s a list of additional things that I did differently:

* Removed all English stopwords from NLTK

* Concatenate all the notes (instead of only using the last discharge summary)

* Performed lemmatization

* Readmission can only be counted once

## Next Steps —> Deep Learning

In part 2 of this project I will apply a Deep Learning transformer model to see if that will further improve my outcome. Stay tuned…
