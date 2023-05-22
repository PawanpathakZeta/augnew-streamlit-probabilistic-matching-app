import streamlit as st
import pandas as pd
import numpy as np    
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import os
import sys
from PIL import Image
from matplotlib import pyplot as plt
import time
import datetime
from io import BytesIO
import uuid
from pyforest import *
import recordlinkage as rl
from recordlinkage.preprocessing import clean
from tqdm import tqdm as tdm
import spacy
from kneed import KneeLocator, DataGenerator as dg



# 1- Main Window -- Layout Settings------------------------------------------------------------
st.set_page_config(layout="wide")
primaryColor="#CA007D" #pink
secondaryColor="#0E47D4" #dark_blue
tertiaryColor ="#25cccf"
light_pink = "#CDC9FA"
backgroundColor="#010046" #navy
secondaryBackgroundColor="#012E71" #dark_blue2 sidebar

# Side bar options - makes it wider
st.markdown(
"""
<style>
[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
width: 450px;
}
[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
width: 5450px00px;
margin-left: -450px;
}
</style>
""",
unsafe_allow_html=True
)

# Example how o use html 
# col3.markdown(f'<code style="color:Lightgreen;font-size:16px;">{"Format Check Completed"}</code> ', unsafe_allow_html=True)

# -----------------------------------------------------------------------------------------------
# 2- Sidebar -- Parameter Settings---------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

# Input Data 
st.sidebar.title("Input Data")


## Unmatched file: this are unknown customers 
unmatched_file = st.sidebar.file_uploader('Unmatched Dataset', type='csv', help='Dataset without email address')

unmatched_file_valid_flag = False
if unmatched_file is not None:
    # Check MIME type of the uploaded file
    if  unmatched_file.name == "unmatched_data.csv":
        unmatched_df = pd.read_csv(unmatched_file)
        unmatched_file_valid_flag = True
        st.sidebar.success('Success')
    else:
        st.sidebar.error('Error: please, re-upload file called unmatched_data.csv')

## Customer file: this are known customers (a.k.a customer databse with PII)
customer_file = st.sidebar.file_uploader("Customer Dataset", type='csv', help='Dataset containing email address')

customer_file_valid_flag = False
if customer_file is not None:
    # Check MIME type of the uploaded file
    if  customer_file.name == "customer_data.csv":
        customer_df = pd.read_csv(customer_file)
        customer_file_valid_flag = True
        st.sidebar.success('Success')
    else:
        st.sidebar.error('Error: please, re-upload file called customer_data.csv')


## Check Data Formats button
check_data = st.sidebar.button("""Check Data""", help = 'Show statistics of your data')


# Persist  unm/cust data statistics
if 'valid_flag' not in st.session_state:
    st.session_state.valid_flag = False

if 'unmatched_obj_cols' not in st.session_state:
    st.session_state.unmatched_obj_cols = ['']
if 'unmatched_num_cols' not in st.session_state:
    st.session_state.unmatched_num_cols = ['']
if 'unmatched_obj_cols_optional' not in st.session_state:
    st.session_state.unmatched_obj_cols_optional = ['']

if 'unmatched_df_com_cols' not in st.session_state:
    st.session_state.unmatched_df_com_cols = pd.DataFrame()
if 'customer_df_com_cols' not in st.session_state:
    st.session_state.customer_df_com_cols = pd.DataFrame()

if check_data:
    if (unmatched_file_valid_flag == True) and (customer_file_valid_flag ==True):
        unmatched_df.columns = map(str.upper, unmatched_df.columns)
        customer_df.columns = map(str.upper, customer_df.columns)  
        commun_cols_u_c = list(unmatched_df.columns[unmatched_df.columns.isin(customer_df.columns)])

        if commun_cols_u_c:
            unmatched_df = unmatched_df[commun_cols_u_c]
            customer_df = customer_df[commun_cols_u_c]
            unmatched_obj_cols = list(unmatched_df.select_dtypes(include="object").columns)
            unmatched_num_cols = list(unmatched_df.select_dtypes(exclude="object").columns)
            customer_obj_cols = list(customer_df.select_dtypes(include="object").columns) 
            customer_num_cols = list(customer_df.select_dtypes(exclude="object").columns)
            assert (unmatched_obj_cols == customer_obj_cols),"String column data types do not match"
            assert (unmatched_num_cols == customer_num_cols),"Numeric column data types do not match"
            st.session_state.valid_flag = True

            unmatched_num_cols.insert(0,"")
            unmatched_obj_cols_optional = unmatched_obj_cols.copy()
            unmatched_obj_cols_optional.insert(0,"")

            st.session_state.unmatched_obj_cols = unmatched_obj_cols
            st.session_state.unmatched_num_cols = unmatched_num_cols
            st.session_state.unmatched_obj_cols_optional = unmatched_obj_cols_optional

            st.session_state.unmatched_df_com_cols = unmatched_df[commun_cols_u_c]
            st.session_state.customer_df_com_cols = customer_df[commun_cols_u_c]

    else:
        pass
else:
    pass



# Parameter Selection 
st.sidebar.title("Parameter Selection")
st.sidebar.subheader('Main field to be matched')

# sidebar columns 5 and 6
col5_sidebar, col6_sidebar= st.sidebar.columns([2, 2])

## Unmatched and Customer columns
if (unmatched_file_valid_flag == True) and (customer_file_valid_flag ==True):
    # Mail field to be matched
    select_box_unmatched_load_main = st.sidebar.selectbox(
        'Select column',
        options=st.session_state.unmatched_obj_cols, 
        help = 'Select main column to apply Match')
    
    ## Main Threshold
    selectbox_threshold = st.sidebar.selectbox('Threshold',
        ('0.95', '0.85', '0.75', '0.60', '0.50'), help='Select main field threshold to perfom match')

    # #Fixed threshold columns 
    st.sidebar.subheader('Run algorithm on')
    select_box_unmatched_load_11 = st.sidebar.selectbox(
        'Select extra column',
        (st.session_state.unmatched_num_cols), 
        help = 'Run algorithm on extra column for 100% threshold if desired'
        , key = 11)
    select_box_unmatched_load_12 = st.sidebar.selectbox(
        'Select extra column',
        (st.session_state.unmatched_num_cols), 
        help = 'Run algorithm on extra column for 100% threshold if desired'
        , key = 12)

    def comp_time_eq(n_rows, threshold):
        if str(threshold) == '0.95':
            z_95 = np.array([ 8.61430182e-04, -1.21567894e+02])
            res = z_95[0] * n_rows + z_95[1]
            if res >= 0:
                return res
            else:
                return 0
        elif str(threshold == '0.75') or str(threshold) == '0.85':
            z_75 = np.array([ 6.15975524e-04, -2.13769878e+01])
            res = z_75[0] * n_rows + z_75[1]
            if res >= 0:
                return res
            else:
                return 0
        elif (str(threshold) == '0.50' or str(threshold) == '0.60') and n_rows < 3000000:
            z_50_a3M = np.array([ 7.43574104e-04, -4.45166486e+01])
            res =  z_50_a3M[0] * n_rows + z_50_a3M[1]
            if res >= 0:
                return res
            else:
                return 0
        elif (str(threshold) == '0.50' or str(threshold) == '0.60') and n_rows >= 3000000:
            z_50_d3M = np.array([ 5.31421923e-03, -1.36647504e+04])
            res = z_50_d3M[0] * n_rows + z_50_d3M[1]
            if res >= 0:
                return res
            else:
                return 0

    x = comp_time_eq(st.session_state['unmatched_df_com_cols'].shape[0], selectbox_threshold)
    if x > 0:
        st.sidebar.markdown(f'<h1 style="color:{tertiaryColor};font-size:16px;">{f"The operation will take {datetime.timedelta(seconds=x)}"}</h1>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f'<h1 style="color:{tertiaryColor};font-size:16px;">{f"The operation will take less than a minute"}</h1>', unsafe_allow_html=True)
    
elif unmatched_file_valid_flag == True:
    selectbox_mss = list([" "])

    # Mail field to be matched
    select_box_unmatched_load_main = st.sidebar.selectbox(
        'Select column',
        options=selectbox_mss, 
        help = 'Upload Customer Database and select columns to apply Match')
    
    selectbox_threshold = st.sidebar.selectbox('Threshold',
        ('0.95', '0.85', '0.75', '0.60', '0.50'), help='Select main field threshold to perfom match')

     # Run algorithm on  
    st.sidebar.subheader('Run algorithm on')
    select_box_unmatched_load_21 = st.sidebar.selectbox(
        'Select column',
        (selectbox_mss), 
        help = 'Upload Customer Database and select columns to apply Match'
        , key = 21)
    select_box_unmatched_load_22 = st.sidebar.selectbox(
        'Select column',
        (selectbox_mss), 
        help = 'Upload Customer Database and select columns to apply Match'
        , key = 22)

elif customer_file_valid_flag == True:
    selectbox_mss = list([" "])
    # Mail field to be matched
    
    select_box_unmatched_load_main = st.sidebar.selectbox(
        'Select column',
        options=selectbox_mss, 
        help = 'Upload Unmatched Dataset and select columns to apply Match')
    
    selectbox_threshold = st.sidebar.selectbox('Threshold',
        ('0.95', '0.85', '0.75', '0.60', '0.50'), help='Select main field threshold to perfom match')

     ## Run algorithm on  
    st.sidebar.subheader('Run algorithm on')
    select_box_unmatched_load_31 = st.sidebar.selectbox(
        'Select column',
        (selectbox_mss), 
        help = 'Upload Unmatched Dataset and select columns to apply Match'
        , key = 31)
    select_box_unmatched_load_32 = st.sidebar.selectbox(
        'Select column',
        (selectbox_mss), 
        help = 'Upload Unmatched Dataset and select columns to apply Match'
        , key = 32)
   
   
else:
    selectbox_mss = list([" "])
    # Mail field to be matched
    select_box_unmatched_main_empty = st.sidebar.selectbox(
        'Select column',
        options=selectbox_mss, help = 'Please, upload data to start')
    ## Main Threshold
    selectbox_threshold = st.sidebar.selectbox('Threshold',
        ('0.95', '0.85', '0.75', '0.60', '0.50'), help='Select main field threshold to perfom match')
    # Run algorithm on
    st.sidebar.subheader('Run algorithm on')
    select_box_unmatched_empty1 =st.sidebar.selectbox(
        """Select extra column"""
        , options=selectbox_mss , help='Please, upload data to start', key = 11)
    select_box_customer_empty1 = st.sidebar.selectbox(
        'Select extra column'
        , options=selectbox_mss , help='Please, upload data to start', key = 12)
    st.sidebar.write(' ')

## Apply Match button
st.sidebar.title("Match data")
match_button = st.sidebar.button("""Apply Match""", help='Apply match to your data and show the results')




# -----------------------------------------------------------------------------------------------
# 1- Main Window -- Parameter Settings-----------------------------------------------------------
# ----------------------------------------------------------------------------------------------- 
# st.title(f'My first app {st.__version__}')


# Creating columns 1 and 2
col1, col2 = st.columns([13, 2])

## Zeta Logo
zeta_logo = Image.open('ZETA_BIG-99e027c9.webp')
col2.image(zeta_logo)

## Header
col1.title("Zeta Customer Matcher")
"""This app demonstrates Customers Probabilistic Matching Project"""

# Creating columns 3 and 4
col_space1, col_space2  =  st.columns([2,2])

# Creating columns 3 and 4
col3, col4= st.columns([2, 2])
colna1, colna2 = st.columns([2,2]) # NA message if no NAs

# Create col expand data
col_left_expander, col_right_expander = st.columns(2)

# Creating columns 5 and 6

col5, col6 = st.columns([2,2])


## Summary pie charts
colp1, colp2, colp3 = st.columns([2, 2, 2])


# Set num of column algorithm will compare, by default is 1
col_compare_alg = 1




# Display features if data is valid and match button is clicked
if match_button and unmatched_file_valid_flag and customer_file_valid_flag:
    if (select_box_unmatched_load_11 == select_box_unmatched_load_12 )and ((select_box_unmatched_load_11 != '') or select_box_unmatched_load_12):
        st.error('Please, select two different extra columns to perform match')
    else:
        if select_box_unmatched_load_main and (select_box_unmatched_load_11 or select_box_unmatched_load_12):
            col_compare_alg = 2 
        if select_box_unmatched_load_main and select_box_unmatched_load_11 and select_box_unmatched_load_12:
            col_compare_alg = 3
        
        # Bar progress
        latest_iteration = st.empty()
        bar = st.progress(0)
        latest_iteration.text('Matching in progress...')


        #------ THIS IS DATA CLEANING

        unmatched_df.drop_duplicates(inplace=True)
        customer_df.drop_duplicates(inplace=True)

        unmatched_df.insert(0,"ID",range(1,unmatched_df.shape[0]+1) ,True)
        customer_df.insert(0,"ID",range(1,customer_df.shape[0]+1) ,True)
        #------ 

        ## -> Data Cleaning start
        ### 1. unmatched_df

        # drop columns with 100% nan values
        unmatched_df = unmatched_df.dropna(axis=1, how='all')

        # get lists of numerical and categorical columns
        numeric_cols = unmatched_df.select_dtypes(include=[np.number]).columns.to_list()
        categorical_cols = unmatched_df.select_dtypes(exclude=[np.number]).columns.to_list()

        def isnumber(x):
                            try:
                                int(x)
                                return int(x)
                            except:
                                return np.nan

        # clean zip codes
        for col in unmatched_df.columns:
            if 'zip' in col.lower():
                if col in categorical_cols:
                    unmatched_df[col] = unmatched_df[col].apply(lambda x: x.replace('.0',''))
                else:
                    unmatched_df[col] = unmatched_df[col].apply(isnumber)
                    unmatched_df[col] = unmatched_df[col].astype(float)
                    unmatched_df[col] = unmatched_df[col].replace(np.nan, 0)
                    unmatched_df[col] = unmatched_df[col].astype('int')

        # clean left and right spaces in categorical columns
        for col in categorical_cols:
            unmatched_df[col]=unmatched_df[col].str.rstrip(" ").str.rstrip(" ").str.strip(" ")

        # avoid double space in between words in categorical columns
        for col in categorical_cols:
            unmatched_df[col] = unmatched_df[col].apply(lambda x: ' '.join(str(x).split())).replace('nan',np.nan)

        ## STANDARDIZE

        # make col names capital
        unmatched_df.columns = unmatched_df.columns.str.upper()
        # make values lower case
        unmatched_df = unmatched_df.applymap(lambda s: s.lower() if type(s) == str else s)

        ## CLEAN EMPTY CELLS AND VALUES WITH SYMBOLS

        # replace empty cells by nan; digits and symbol "-" are allowed; email column is excluded
        for col in categorical_cols:
            #if 'EMAIL' not in col:
            unmatched_df[col] = unmatched_df[col].replace('', np.nan).replace(' ', np.nan)
        
        ## AUTOMATIC DETECTION OF PERSON NAME COLUMNS

        # select columns that have at least 10 notNa values
        mask = ( unmatched_df.shape[0] - unmatched_df.isna().sum(axis=0) ) > 9
        cols_to_check_for_names = unmatched_df.columns[mask].to_list()

        # load English tokenizer, tagger, parser and NER
        nlp = spacy.load("en_core_web_sm")

        # identify columns that contain person names
        name_cols = [] 

        for col in cols_to_check_for_names:
            # get 10 values in column
            el = unmatched_df[unmatched_df[col].notnull()][col] # pick values that are notNa
            el = np.random.permutation(el) # shuffle the values
            el = el[:10].tolist() # pick 10 first elements and convert to list
            
            # create a string with sequence of 10 first values
            string = ''
            for word in el:
                try: # try method to avoid error for numerical columns
                    string += word + ' '
                except:
                    pass
            
            # check how many string elements are person names
            if len(string) > 0:
                isname = nlp(string)
                names_list = [token for token in isname if token.ent_type_ == 'PERSON']
                if len(names_list) > 3: # add column if at least 5 names (we're checking 10) were found
                    name_cols.append(col)

        name_cols = [x for x in name_cols if 'user' not in x.lower() and 'email' not in x.lower() and 'city' not in x.lower()]

        if len(name_cols) == 0:
            name_cols = [col for col in unmatched_df.columns if 'name' in col.lower()]
            name_cols = [x for x in name_cols if 'user' not in x.lower() and 'email' not in x.lower() and 'city' not in x.lower()]

        ## ELIMINATE SYMBOLS AND INPUT NAN IN CELLS THAT CONTAIN DIGITS

        for col in name_cols:
            unmatched_df[col] = unmatched_df[col].str.replace(r'[^\w\s-]+', '').str.replace('_','') # eliminate symbols
            unmatched_df[col] = unmatched_df[unmatched_df[col].str.contains(r"^[a-zA-Z\s-]*$", np.nan, regex=True, na=False)][col] # nan if contains digit

        ## REMOVE BAD NAMES FROM PERSON NAME COLUMNS

        # set wildcard with words that shouldn't be included in name cols
        wildcard = set(['credit','debit','card', 'debito','credito','mastercard', 'visa', 'customer', 'value', 'cardmember'])  

        for col in name_cols:
            unmatched_df[col].fillna('-99',inplace=True) # input -99 to nan values
            unmatched_df['words_found'] = [','.join(sorted(wildcard.intersection(l))) 
                                    for l in unmatched_df[col].str.lower().str.split()]

            unmatched_df['found'] = unmatched_df['words_found'].astype(bool).astype(int)
            unmatched_df = unmatched_df[unmatched_df['found'] == 0]

            # drop additional columns
            unmatched_df.drop(['words_found','found'],axis=1, inplace=True)
            
            # input nan back to missing values
            unmatched_df[col] = unmatched_df[col].replace('-99', np.nan)

        ## REMOVE NaNs FROM SELECTED COLS TO JOIN ON
        try: # try to avoid error if select box left blank
            unmatched_df = unmatched_df[unmatched_df[select_box_unmatched_load_main].notnull()]
        except:
            pass
        try:
            unmatched_df = unmatched_df[unmatched_df[select_box_unmatched_load_11].notnull()]
        except:
            pass
        try:
            unmatched_df = unmatched_df[unmatched_df[select_box_unmatched_load_12].notnull()]
        except:
            pass

        ### 2. customer_df

        # drop columns with 100% nan values
        customer_df = customer_df.dropna(axis=1, how='all')

        # get lists of numerical and categorical columns
        numeric_cols = customer_df.select_dtypes(include=[np.number]).columns.to_list()
        categorical_cols = customer_df.select_dtypes(exclude=[np.number]).columns.to_list()
        
        # Clean zip codes
        for col in customer_df.columns:
            if 'zip' in col.lower():
                if col in categorical_cols:
                    customer_df[col] = customer_df[col].apply(lambda x: x.replace('.0',''))
                else:
                    customer_df[col] = customer_df[col].apply(isnumber)
                    customer_df[col] = customer_df[col].astype(float)
                    customer_df[col] = customer_df[col].replace(np.nan, 0)
                    customer_df[col] = customer_df[col].astype('int')
        
        # clean left and right spaces in categorical columns
        for col in categorical_cols:
            customer_df[col]=customer_df[col].str.rstrip(" ").str.rstrip(" ").str.strip(" ")

        # avoid double space in between words in categorical columns
        for col in categorical_cols:
            customer_df[col] = customer_df[col].apply(lambda x: ' '.join(str(x).split())).replace('nan',np.nan)

        ## STANDARDIZE

        # make col names capital
        customer_df.columns = customer_df.columns.str.upper()
        # make values lower case
        customer_df = customer_df.applymap(lambda s: s.lower() if type(s) == str else s)

        ## CLEAN EMPTY CELLS AND VALUES WITH SYMBOLS

        # replace empty cells and symbols by nan; digits and symbol "-" are allowed; email column is excluded
        for col in categorical_cols:
            #if 'EMAIL' not in col:
            customer_df[col] = customer_df[col].replace('', np.nan).replace(' ', np.nan)

        ## AUTOMATIC DETECTION OF PERSON NAME COLUMNS

        # select columns that have at least 10 notNa values
        mask = ( customer_df.shape[0] - customer_df.isna().sum(axis=0) ) > 9
        cols_to_check_for_names = customer_df.columns[mask].to_list()

        # identify columns that contain person names
        name_cols = [] 

        for col in cols_to_check_for_names:
            # get 10 values in column
            el = customer_df[customer_df[col].notnull()][col] # pick values that are notNa
            el = np.random.permutation(el) # shuffle the values
            el = el[:10].tolist() # pick 10 first elements and convert to list
            
            # create a string with sequence of 10 first values
            string = ''
            for word in el:
                try: # try method to avoid error for numerical columns
                    string += word + ' '
                except:
                    pass
            
            # check how many string elements are person names
            if len(string) > 0:
                isname = nlp(string)
                names_list = [token for token in isname if token.ent_type_ == 'PERSON']
                if len(names_list) > 3: # add column if at least 5 names (we're checking 10) were found
                    name_cols.append(col)

        name_cols = [x for x in name_cols if 'user' not in x.lower() and 'email' not in x.lower() and 'city' not in x.lower()]

        if len(name_cols) == 0:
            name_cols = [col for col in customer_df.columns if 'name' in col.lower()]
            name_cols = [x for x in name_cols if 'user' not in x.lower() and 'email' not in x.lower() and 'city' not in x.lower()]

        ## ELIMINATE SYMBOLS AND INPUT NAN IN CELLS THAT CONTAIN DIGITS

        for col in name_cols:
            #customer_df[col]=customer_df[customer_df[col].str.contains(r"^[a-zA-Z\s-]*$", np.nan, regex=True, na=False)][col]
            customer_df[col] = customer_df[col].str.replace(r'[^\w\s-]+', '').str.replace('_','') # eliminate symbols
            customer_df[col] = customer_df[customer_df[col].str.contains(r"^[a-zA-Z\s-]*$", np.nan, regex=True, na=False)][col] # nan if contains digit


        ## REMOVE BAD NAMES FROM PERSON NAME COLUMNS


        for col in name_cols:
            customer_df[col].fillna('-99',inplace=True) # input -99 to nan values
            customer_df['words_found'] = [','.join(sorted(wildcard.intersection(l))) 
                            for l in customer_df[col].str.lower().str.split()]

            customer_df['found'] = customer_df['words_found'].astype(bool).astype(int)
    
            #print(customer_df.shape)
            customer_df = customer_df[customer_df['found'] == 0]
            #print(customer_df.shape)
        
            # drop additional columns
            customer_df.drop(['words_found','found'],axis=1, inplace=True)
    
            # input nan back to missing values
            customer_df[col] = customer_df[col].replace('-99', np.nan)

        ## CLEAN ZIPCODES

        # input NaN in value is not numeric, then convert column type to float (int doesn't allow NaNs)

        customer_df.drop_duplicates(inplace=True)
        unmatched_df.drop_duplicates(inplace=True)

    ## <- Data Cleaning end

        #latest_iteration = st.empty()
        bar.progress((100//4)*1)
        latest_iteration.text('Initiating algorithm...')

        # ALGORITHM FUNTIONS

        def cleaning_cols(df):
            for  col in df.select_dtypes("object").columns:
                df[col]=clean(df[col])
            return df

        def Sorted_Neighbourhood_Prediction(df1,df2, pred_comp=col_compare_alg, threshold=None,method_str=None,method_num=None,scale=None,offset=None, main_field_compare=None):
            #cleaning object cols for model redeability
            df1=cleaning_cols(df1)
            df2=cleaning_cols(df2)
            threshold =float(threshold)
                
            #resetiing index to core customerids of respective datasets
            df1=df1.set_index('ID')
            df2=df2.set_index('ID')
            
            ## creating mathced indexes using SoretdNeighbourHood Approach
            clx = rl.index.SortedNeighbourhood(main_field_compare, window=5)
            clx = clx.index(df1, df2)

            cr = rl.Compare()
            cr.string(main_field_compare, main_field_compare,method=method_str,threshold=threshold, label=main_field_compare)
            if select_box_unmatched_load_11:
                cr.numeric(select_box_unmatched_load_11, select_box_unmatched_load_11,scale=scale, offset=offset, label=select_box_unmatched_load_11)
            if select_box_unmatched_load_12:
                cr.numeric(select_box_unmatched_load_12, select_box_unmatched_load_12, method=method_num, scale=scale, offset=offset, label=select_box_unmatched_load_12)

            feature_vectors = cr.compute(clx,df1,df2)
                
            predictions =feature_vectors[feature_vectors.sum(axis=1) > round(threshold*pred_comp,1)]
            
            return feature_vectors,predictions
        #latest_iteration = st.empty()
        bar.progress((100//4)*2)
        latest_iteration.text('Sorting predictions...')

        model_thld = float(selectbox_threshold)

        vectors,predictions= Sorted_Neighbourhood_Prediction(unmatched_df,
                                                            customer_df,
                                                            pred_comp =col_compare_alg,
                                                            threshold=model_thld,
                                                            method_str="levenshtein",
                                                            method_num="step",
                                                            scale=5,
                                                            offset=5,
                                                            main_field_compare = select_box_unmatched_load_main)
        #latest_iteration = st.empty()
        bar.progress((100//4)*3)
        latest_iteration.text('Calculating number of matches...')

        data_indexes=predictions.reset_index()
        data_indexes=data_indexes["ID_1	ID_2".split("\t")]
        df_v1=data_indexes.merge(unmatched_df,left_on="ID_1",right_on="ID")
        df_v2=data_indexes.merge(customer_df,left_on="ID_2",right_on="ID")
        df_final = df_v1.merge(df_v2,on=["ID_1","ID_2"],how="left",suffixes=('_MTCH', '_UNMTCH'))
        # drop duplicate id
        df_final.drop(["ID_1", "ID_2"],axis=1,inplace=True)
        #drop duplicates
        df_final.drop_duplicates(inplace=True)

        # Calculate number of matches for all thresholds
        # This will help us building elbow plot and the table below
        threshold_possibilities = ['0.50','0.60','0.75','0.85','0.90','0.95']
        count_num_mtch = []

        for possibility in threshold_possibilities:
            vectors,predictions= Sorted_Neighbourhood_Prediction(unmatched_df,
                                                                customer_df,
                                                                pred_comp =col_compare_alg,
                                                                threshold=possibility,
                                                                method_str="levenshtein",
                                                                method_num="step",
                                                                scale=5,
                                                                offset=5,
                                                                main_field_compare = select_box_unmatched_load_main)
            data_indexes=predictions.reset_index()
            data_indexes=data_indexes["ID_1	ID_2".split("\t")]
            df_v1=data_indexes.merge(unmatched_df,left_on="ID_1",right_on="ID")
            df_v2=data_indexes.merge(customer_df,left_on="ID_2",right_on="ID")
            df_final_all_trhld = df_v1.merge(df_v2,on=["ID_1","ID_2"],how="left",suffixes=('_MTCH', '_UNMTCH'))
            # drop duplicate id
            df_final_all_trhld.drop(["ID_1", "ID_2"],axis=1,inplace=True)
            #drop duplicates
            df_final_all_trhld.drop_duplicates(inplace=True)

            # count number of matches per threshold 
            count_num_mtch.append(len(df_final_all_trhld))



        # Create threshold vs #matches table

        match_dict = dict(zip(threshold_possibilities, count_num_mtch))
        data = {'Threshold': match_dict.keys(), '#Match': match_dict.values()}
        best_threshold_df = pd.DataFrame.from_dict(data, orient='index')
        best_threshold_df.columns = [''] * len(best_threshold_df.columns)
        best_threshold_df_T = best_threshold_df.copy() #Save here best_threshold_df transpose to use it later for a graph
        new_header = best_threshold_df.iloc[0] #grab the first row for the header
        best_threshold_df = best_threshold_df[1:] #take the data less the header row
        best_threshold_df.columns = new_header
        best_threshold_df = best_threshold_df.reset_index().rename(columns={'index':'Threshold'})

        #best_threshold_df_T transpose to use it later for elbow graph
        best_threshold_df_T = pd.DataFrame(best_threshold_df_T, index = ['Threshold','#Match']).transpose() 
        best_threshold_df_T.Threshold =  best_threshold_df_T.Threshold.astype(float)
        best_threshold_df_T['#Match'] = best_threshold_df_T['#Match'].astype(float)

        # /....
        ## Build Elbow line chart: Numb Match vs Threshold
        x_threshold = best_threshold_df_T['Threshold']
        y_num_match = best_threshold_df_T['#Match']

        elbow_chart, ax = plt.subplots()
        elbow_chart.patch.set_facecolor(backgroundColor)
        ax.set_facecolor(backgroundColor)
        ax.plot(x_threshold, y_num_match, secondaryColor, linewidth = 2.5, marker= 'o')
        plt.fill_between(x_threshold, y_num_match, color=primaryColor, alpha=0.7)

        # Giving x label using xlabel() method
        # with bold setting
        plt.xlabel("Threshold", color='white',fontsize =14)
        plt.tick_params(colors='white', which='both') 
        # Y label settings, use ontweight='bold' to make font bold
        plt.ylabel("# Matches", color='white', rotation='vertical', loc ='center', fontsize =14)
        plt.ylim(((min(y_num_match)-50),(max(y_num_match)+50)))
        # Giving title to the plotTotal customers 100% matc
        plt.title("Number Matched Records per Threshold", color='white', fontsize =16, y=1.1)
        # Add cosmetics
        ax.spines['bottom'].set_color('lightgrey')
        ax.spines['top'].set_color('lightgrey') 
        ax.spines['right'].set_color('lightgrey')
        ax.spines['left'].set_color('lightgrey')
        # Showing the plot using plt.show()
        plt.show()
        # /....

        # Calculate Optimal threshold
        x_threshold = best_threshold_df_T['Threshold']
        y_num_match = best_threshold_df_T['#Match']
        x_threshold, y_num_match = dg.convex_increasing()
        kl = KneeLocator(x_threshold, y_num_match, curve="convex")
        best_threshold =kl.knee/10
        def select_optimal_elbow_point(best_threshold):
            if best_threshold <0.55:
                return 0.5
            elif (best_threshold >=0.55) & (best_threshold <0.63):
                return 0.6
            elif (best_threshold >=0.63) & (best_threshold <0.79):
                return 0.75
            elif (best_threshold >=0.79) & (best_threshold <0.87):
                return 0.85
            elif (best_threshold >=0.87) & (best_threshold <0.95):
                return 0.9
            elif (best_threshold >=0.93):
                return 0.95    

        select_optimal_elbow_point(best_threshold)

        ## Example Fuzzy Logic output table 
        algrtm_output_df = df_final.copy()
        
        if select_box_unmatched_load_main and select_box_unmatched_load_11 and select_box_unmatched_load_12:     
            algrtm_output_df = algrtm_output_df[[f'{select_box_unmatched_load_main}_MTCH', 
                                                    f'{select_box_unmatched_load_main}_UNMTCH', 
                                                    f'{select_box_unmatched_load_11}_MTCH', 
                                                    f'{select_box_unmatched_load_11}_UNMTCH',
                                                    f'{select_box_unmatched_load_12}_MTCH',  
                                                    f'{select_box_unmatched_load_12}_UNMTCH']]

        elif select_box_unmatched_load_main and select_box_unmatched_load_11:
            algrtm_output_df = algrtm_output_df[[f'{select_box_unmatched_load_main}_MTCH', 
                                                    f'{select_box_unmatched_load_main}_UNMTCH', 
                                                    f'{select_box_unmatched_load_11}_MTCH', 
                                                    f'{select_box_unmatched_load_11}_UNMTCH']]

        elif select_box_unmatched_load_main and select_box_unmatched_load_12:
            algrtm_output_df = algrtm_output_df[[f'{select_box_unmatched_load_main}_MTCH', 
                                                    f'{select_box_unmatched_load_main}_UNMTCH', 
                                                    f'{select_box_unmatched_load_12}_MTCH', 
                                                    f'{select_box_unmatched_load_12}_UNMTCH']]

        elif select_box_unmatched_load_main:
            algrtm_output_df = algrtm_output_df[[f'{select_box_unmatched_load_main}_MTCH', 
                                                    f'{select_box_unmatched_load_main}_UNMTCH']]

        # To do: in the future replace here the main column that we match (MSRNAME_MTCH) with a variable --Andrea
        df_copy = df_final.copy()
        df_copy['100_match'] = np.where(df_final.MSRNAME_MTCH == df_final.MSRNAME_UNMTCH, 1, 0)
        df_copy['100_match_address'] = np.where(df_copy.STORE_CITY_MTCH == df_copy.STORE_CITY_UNMTCH, 1, 0)

        df_gb_store_mtch = pd.DataFrame()
        # Group by Store --- Name matching
        df_gb_store_mtch['TOTAL_CUSTOMERS'] = df_copy.groupby('STORE_MTCH')['STORE_MTCH'].count()
        df_gb_store_mtch['100%_MATCH'] =df_copy.groupby('STORE_MTCH')['100_match'].sum()
        df_gb_store_mtch['MISSPELLING'] =  df_gb_store_mtch['TOTAL_CUSTOMERS']  - df_gb_store_mtch['100%_MATCH']
        df_gb_store_mtch['MISSPELLING_%'] =  np.round(df_gb_store_mtch['MISSPELLING'] * 100 / df_gb_store_mtch['TOTAL_CUSTOMERS'], 2)

        df_gb_store_mtch = df_gb_store_mtch.sort_values('MISSPELLING_%', ascending=False)
        df_gb_store_mtch = df_gb_store_mtch.reset_index().rename(columns={'STORE_MTCH': 'STORE'})
        


        # Col 3 text
        col5.subheader(f'Perform Matching at Threshold: {selectbox_threshold}')
        col5.text(f'Optimal Threshold: {select_optimal_elbow_point(best_threshold)} (based on number of matches vs quality)')
        col5.subheader(' ')

        #latest_iteration = st.empty()
        bar.progress((100//4)*4)
        latest_iteration.text('Process completed!')

        # Display Elbow chart
        col5.pyplot(elbow_chart)

        #Display best_threshold_df table
        styler = best_threshold_df.style.hide_index()
        col5.write(styler.to_html(), unsafe_allow_html=True) 

        # /....
        # Calculations to display Match Venn Graph
        match_count = np.round(match_dict[selectbox_threshold], 1)
        unmatched_df_count = np.round((len(st.session_state['unmatched_df_com_cols']) - match_count), 1)
        matched_df_count = np.round((len(st.session_state.customer_df_com_cols) - match_count), 1)

        # Match Venn Graph 
        fig, ax = plt.subplots()
        fig.patch.set_facecolor(backgroundColor)

        plt.title('Matching', color = 'white', fontsize=14)
        plt.figure(figsize=(10,10))

        total= 100.0
        v = venn2(subsets=(unmatched_df_count, matched_df_count, match_count), 
                    set_labels=("Unmatched Records", "Customer Database"), 
                    # subset_label_formatter=lambda x: f"{(np.round(x/1000000, 1))}M", #uncomment if you want to round you len(df) to Millions
                    subset_label_formatter=lambda x: f"{(np.round(x/1000, 1))}K",
                    ax=ax,set_colors=(primaryColor, secondaryColor), alpha = 0.7)
        i = 0
        for text in v.set_labels:
            text.set_color('white')
            text.set_fontsize(14) 
            i+=1

        for text in v.subset_labels:   
            text.set_color('white')   
            text.set_fontsize(14)   
            # text.set_fontweight('bold')
        plt.show()
        
        
        # Display Match Venn Graph ONLY if unmatched and customer .csv are uploaded and the threshold was set
        col6.subheader(f'')
        col6.text(f'')
        col6.subheader(' ')
        col6.subheader(' ')
        col6.subheader(' ')
        col6.subheader(' ')
        col6.pyplot(fig)
        # /....

        bar.empty()
        latest_iteration.empty()

        # Display dataset summary after Probabilistic Matching
        st.subheader('Dataset Matched Summary​')


        # /....
        #Autocorrect Statistics ----------------------------------------
        unmtch_name_col = st.session_state['unmatched_df_com_cols'].MSRNAME


        ctomer_name_col = st.session_state.customer_df_com_cols.MSRNAME

        df_final['cmplete_name_match'] = np.where(df_final[f'{select_box_unmatched_load_main}_MTCH'] == df_final[f'{select_box_unmatched_load_main}_UNMTCH'], 1, 0)

        cmplete_name_match= len(df_final[df_final['cmplete_name_match']==1])
        mtch_output_len = len(df_final)
        cmplete_name_match_percentage = np.round(cmplete_name_match*100/mtch_output_len, 2)

        ## Summary pie charts
        colp1, colp2, colp3 = st.columns([2, 2, 2])
    
        # pie chart 1
        y = np.array([unmtch_name_col.nunique(), len(unmtch_name_col) - unmtch_name_col.nunique()])
        labels = ['Unmatched unique customers', 'Duplicates']
        mycolors = [secondaryColor, primaryColor]
        explode = (0, 0)
    
        #fig1, ax1 = plt.subplots()
        fig1, ax1 = plt.subplots(figsize=(6,6))
        fig1.patch.set_facecolor(backgroundColor)
    
        ax1.pie(y, autopct=lambda x: '{:.0f}'.format(x*y.sum()/100), explode=explode, labels=None,
            shadow=True, startangle=90, colors=mycolors, textprops={'color':"w",'fontsize': 14}, wedgeprops={'alpha':0.7}) 
        ax1.axis('equal')
        plt.title('Unmatched dataset', color='white', y=1.1, fontsize=16)
        plt.legend(labels, loc='lower left', bbox_to_anchor=(0.5, -0.08))
        #plt.figure(figsize=(10,10))

        buf = BytesIO()
        fig1.savefig(buf, format="png")
        colp1.image(buf)   

        #colp1.pyplot(fig1)
    
        # pie chart 2
        y = np.array([ctomer_name_col.nunique(), len(ctomer_name_col) - ctomer_name_col.nunique()])
        labels = ['Known unique customers', 'Duplicates']
        mycolors = [secondaryColor, primaryColor]
        explode = (0, 0)
    
        #fig2, ax2 = plt.subplots()
        fig2, ax2 = plt.subplots(figsize=(6,6))
        fig2.patch.set_facecolor(backgroundColor)
    
        ax2.pie(y, autopct=lambda x: '{:.0f}'.format(x*y.sum()/100), explode=explode, labels=None,
            shadow=True, startangle=90, colors=mycolors, textprops={'color':"w",'fontsize': 14}, wedgeprops={'alpha':0.7}, radius=1800)
        ax2.axis('equal')
        plt.title('Customer dataset', color='white', y=1.1, fontsize=16)
        plt.legend(labels, loc='lower left', bbox_to_anchor=(0.5, -0.08))
        #plt.figure(figsize=(10,10))
    
        buf = BytesIO()
        fig2.savefig(buf, format="png")
        colp2.image(buf) 

        #colp2.pyplot(fig2)
    
        # pie chart 3
        y = np.array([len(df_final['cmplete_name_match']) - cmplete_name_match, cmplete_name_match])
        labels = ['Not exact match', '100% exact match']
        mycolors = [secondaryColor, primaryColor]
        explode = (0, 0)
    
        #fig3, ax3 = plt.subplots()
        fig3, ax3 = plt.subplots(figsize=(6,6))
        fig3.patch.set_facecolor(backgroundColor)
    
    
        ax3.pie(y, autopct=lambda x: '{:.0f}'.format(x*y.sum()/100), explode=explode, labels=None,
            shadow=True, startangle=90, colors=mycolors, textprops={'color':"w",'fontsize': 14}, wedgeprops={'alpha':0.7}, radius=1800)
        ax3.axis('equal')
        plt.title('Threshold dataset', color='white', y=1.1, fontsize=16)
        plt.legend(labels, loc='lower left', bbox_to_anchor=(0.5, -0.08))
        #plt.figure(figsize=(10,10))

        buf = BytesIO()
        fig3.savefig(buf, format="png")
        colp3.image(buf) 
        #colp3.pyplot(fig3)

        with colp1:
            st.write('')
        with colp2:
            st.write('')
        with colp3:
            st.write('')

        st.dataframe(data=algrtm_output_df.head(100), width=None, height=None)

        st.subheader('Store Performance​')

        #----- Col 7 and Col 8--------
        store_table_title = '<p style="color:White; font-size: 18px; text-align: left;">Store Statistics Table</p>'
        st.markdown(store_table_title, unsafe_allow_html=True)
        st.dataframe(data=df_gb_store_mtch)
        st.subheader(' ')
    
        # /....
        col7,  col8 = st.columns([2, 2])

        # /....
        # Violin plot
        violin_plot = df_gb_store_mtch[['TOTAL_CUSTOMERS', 'MISSPELLING']]
        ax_violin_plot = plt.figure()
        ax_violin_plot.patch.set_facecolor(backgroundColor)
        ax_violin_plot
        ax = ax_violin_plot.add_axes([0,0,1,1])
        ax.set_facecolor('#010046')
        plt.tick_params(colors='white', which='both')
        ax.tick_params(axis='x', colors=backgroundColor)


        parts = ax.violinplot(violin_plot)

        # now change colors
        for pc in parts['bodies']:
            pc.set_facecolor(primaryColor)
            pc.set_edgecolor(secondaryColor)
            pc.set_alpha(1)

        #plt.grid(color = 'lightgrey', linestyle = '--', linewidth = 0.75)
        ax.spines['bottom'].set_color('lightgrey')
        ax.spines['top'].set_color('lightgrey')
        ax.spines['right'].set_color('lightgrey')
        ax.spines['left'].set_color('lightgrey')
        plt.xlabel("Total Customers                                      Misspelling    ",fontsize=14,color='white')
        plt.ylabel("# Customers per store",  color='white', rotation='vertical', loc ='center',fontsize=14)
        plt.title("Store Statistics Distribution", color='white', fontsize=16, y=1.1)
        plt.figure(figsize=(10,10))

        col7.pyplot(ax_violin_plot)

        # /....
    
elif match_button and (unmatched_file_valid_flag or customer_file_valid_flag):
    col_space1.error('Please, enter valid data and click "Apply Match" to perform matching')
elif unmatched_file_valid_flag and customer_file_valid_flag:
    col_space1.info('Please, check data and apply match')
    st.empty()
else:
    col_space1.info('Please, upload Unmatched Dataset and Customer Dataset to start')
    col_space1.empty()


if st.session_state['valid_flag']:
    #If both files are uploaded print stats of each one
    if  unmatched_file_valid_flag and customer_file_valid_flag:
        col_space1.success('Format Check Completed')    
        df_info = {'Number of rows':[len(st.session_state['unmatched_df_com_cols']), len(st.session_state.customer_df_com_cols)],
                    'Number of columns': [len(st.session_state['unmatched_df_com_cols'].columns), len(st.session_state.customer_df_com_cols.columns)]}
        df_info = pd.DataFrame(df_info).transpose().rename(columns={0:'Unmatch Data', 1:'Customer Data'})
        col_space1.dataframe(df_info)

        ## NA charts
        null_df_unmatched = st.session_state['unmatched_df_com_cols'].apply(lambda x: sum(x.isnull())).to_frame(name='count').reset_index()
        null_df_customers = st.session_state.customer_df_com_cols.apply(lambda x: sum(x.isnull())).to_frame(name='count').reset_index()

        unmatched_na, ax = plt.subplots()
        unmatched_na.patch.set_facecolor(backgroundColor)
        ax.set_facecolor(backgroundColor)
        x = null_df_unmatched['index']
        y = null_df_unmatched['count']
        ax.barh(x, y, color=primaryColor)
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='x', colors='white')
        ax.spines['bottom'].set_color(backgroundColor)
        ax.spines['top'].set_color(backgroundColor)
        ax.spines['right'].set_color(backgroundColor)
        ax.spines['left'].set_color(backgroundColor)
        plt.title("NAs on unmatched data", color='white', fontsize =20, fontweight="bold")
        col3.pyplot(unmatched_na)
        st.write('')

        unmatched_cust, ax = plt.subplots()
        unmatched_cust.patch.set_facecolor(backgroundColor)
        ax.set_facecolor(backgroundColor)
        x = null_df_customers['index']
        y = null_df_customers['count']
        ax.barh(x, y, color=primaryColor)
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='x', colors='white')
        ax.spines['bottom'].set_color(backgroundColor)
        ax.spines['top'].set_color(backgroundColor)
        ax.spines['right'].set_color(backgroundColor)
        ax.spines['left'].set_color(backgroundColor)
        #plt.xticks(null_df_customers.index, null_df_customers.index,
                #  horizontalalignment='right')
        plt.title("NAs on customer data", color='white', fontsize =20, fontweight="bold")
        col4.pyplot(unmatched_cust)
        st.write('')
        
        # Unmatch df
        col_left_expander.write('Unmatched data')
        with col_left_expander.expander("Expand data and statistics"):
            #Display unmatch df
            st.dataframe(st.session_state['unmatched_df_com_cols'].head(100))
            #Display unmatch df NAs 
            st.write(f'Number of Nan values in Unmatched Dataset: ')
            na_unmt_df = pd.DataFrame(st.session_state['unmatched_df_com_cols'].isna().sum())
            na_unmt_df = na_unmt_df.rename(columns={0:'#NAs'})
            st.dataframe(na_unmt_df)
        
        if len(na_unmt_df[na_unmt_df['#NAs']>0]) == 0:
            colna1.info('No NAs found in Unmatched Dataset')

        # Customer df
        col_right_expander.write(f'Customer data')
        with col_right_expander.expander("Expand data and statistics"):
            #Display customer df
            st.dataframe(st.session_state.customer_df_com_cols.head(100))
            #Display unmatch df NAs 
            st.write(f'Number of Nan values in Customer Data: ')
            na_cust_df = pd.DataFrame(st.session_state.customer_df_com_cols.isna().sum())
            na_cust_df = na_cust_df.rename(columns={0:'#NAs'})
            st.dataframe(na_cust_df)
        if len(na_cust_df[na_cust_df['#NAs']>0]) == 0:
            colna2.info('No NAs found in Customer Dataset')


    # Print unmatched_file stats if uploaded
    elif unmatched_file_valid_flag:
        col_space1.success('Format Check Completed')
        col_left_expander.write('Unmatched data')
        with col_left_expander.expander("Expand data and statistics"):
            #Display unmatch df
            st.dataframe(st.session_state['unmatched_df_com_cols'].head(100))
            #Display unmatch df NAs 
            st.write(f'Number of Nan values in Unmatched Dataset: ')
            na_unmt_df = pd.DataFrame(st.session_state['unmatched_df_com_cols'].isna().sum())
            na_unmt_df = na_unmt_df.rename(columns={0:'#NAs'})
            st.dataframe(na_unmt_df)

    # Print customer_file stats if uploaded
    elif customer_file_valid_flag:
        col_space1.success('Format Check Completed')
        col_right_expander.write(f'Customer data')
        with col_right_expander.expander("Expand data and statistics"):
            #Display customer df
            st.dataframe(st.session_state.customer_df_com_cols.head(100))
            #Display unmatch df NAs 
            st.write(f'Number of Nan values in Customer Data: ')
            na_cust_df = pd.DataFrame(st.session_state.customer_df_com_cols.isna().sum())
            na_cust_df = na_cust_df.rename(columns={0:'#NAs'})
            st.dataframe(na_cust_df)

    else: 
         pass
else:
    col3.write('')
