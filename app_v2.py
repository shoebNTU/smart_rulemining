import streamlit as st 
import SessionState
import pandas as pd
import numpy as np
import os
import sys
import json
import time
from datetime import datetime
import subprocess
import threading
from glob import glob
from random import randint
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, apriori, fpmax
from mlxtend.frequent_patterns import association_rules
from datetime import datetime
import base64
import re

import sys


# import visualization tools 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
# import seaborn as sns 

def process_csv(csv_file_buffer, mode):
        df = pd.read_csv(csv_file_buffer)
        # Creating a text element to let the reader know that the data is loading.
        st.text('Loading data...')
        df_replaceNAN = df.replace(np.nan, 'NA', regex=True)
        df_replaceNAN['CMD'] = df_replaceNAN['CMD'].apply(lambda x:str(x).lower())
        df_replaceNAN['PCR'] = df_replaceNAN[['Problem Code','Cause Code', 'Remedy Code']].apply(lambda x:('-').join(x.astype(str)),axis=1)
        processed_dataset = df_replaceNAN[['CMD','PCR']].to_numpy()
        te = TransactionEncoder()
        te_ary = te.fit(processed_dataset).transform(processed_dataset)
        transformed_dataset = pd.DataFrame(te_ary, columns=te.columns_)
        # Notifying the reader that the data was successfully loaded.
        st.text('Loading data...done!')      

        return transformed_dataset,df_replaceNAN

def runApriori(dataset):
        start = datetime.now()
        frequent_itemsets_ap = apriori(dataset,min_support=0.1,use_colnames=True) 
        deltatime = datetime.now() - start
        frequent_itemsets_ap.sort_values(by='support',ascending=False,inplace=True)
        association_rule_ap = association_rules(frequent_itemsets_ap,metric='confidence',min_threshold=0.1)   
        association_rule_ap['length'] = association_rule_ap['consequents'].apply(lambda x: len(x))
        deltatime = datetime.now() - start
        st.success("Completed in {} seconds".format(deltatime.seconds + deltatime.microseconds / 1000000))
        return association_rule_ap

def runFPgrowth(dataset,df_replaceNaN):
        start = datetime.now()
        frequent_itemsets_fp = fpgrowth(dataset,min_support=0.01 ,use_colnames=True) 
        deltatime = datetime.now() - start
        frequent_itemsets_fp.sort_values(by='support',ascending=False,inplace=True)
        association_rule_fp = association_rules(frequent_itemsets_fp,metric='confidence',min_threshold=0.1) 
        association_rule_fp.sort_values(by='leverage',ascending=False,inplace=True)    
        association_rule_fp['length'] = association_rule_fp['consequents'].apply(lambda x: len(x))
        association_rule_fp = association_rule_fp[association_rule_fp.consequents.astype(str).str.contains('|'.join(df_replaceNaN.PCR.unique().tolist()))] 
        deltatime = datetime.now() - start
        st.success("Completed in {} seconds".format(deltatime.seconds + deltatime.microseconds / 1000000))
        return association_rule_fp

def runFPmax(dataset):
        start = datetime.now()
        frequent_itemsets_fpmax = fpmax(dataset,min_support=0.1 ,use_colnames=True) 
        frequent_itemsets_fpmax.sort_values(by='support',ascending=False,inplace=True)
        association_rule_fpmax = association_rules(frequent_itemsets_fpmax,min_threshold=0.1, support_only=True) 
        association_rule_fpmax.sort_values(by='leverage',ascending=False,inplace=True)    
        association_rule_fpmax['length'] = association_rule_fpmax['consequents'].apply(lambda x: len(x))
        deltatime = datetime.now() - start
        st.success("Completed in {} seconds".format(deltatime.seconds + deltatime.microseconds / 1000000))
        return association_rule_fpmax


def process_metrics():
        session_state.support = st.sidebar.slider("support", 0.0, 1.0, 0.3)
        session_state.confidence = st.sidebar.slider("confidence", 0.0, 1.0,0.7)
        session_state.lift = st.sidebar.slider("lift", 0.0, 1000.0,1.2)

        return session_state.support, session_state.confidence, session_state.lift


def initSession():

        sessionstate_init = SessionState.get(

                association_rules = None,
                support	= 0.1,
                confidence = 0.1,
                lift = 0.1	

        )

        return sessionstate_init


if __name__ == "__main__":

        def get_table_download_link(df):
                """Generates a link allowing the data in a given panda dataframe to be downloaded
                in:  dataframe
                out: href string
                """
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                href = f'<a href="data:file/csv;base64,{b64}" download="rules.csv">Download rules as a csv file</a>'
                return href
                # href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
                # return href

        st.sidebar.image("./imgs/siemens_logo.png", width = 300)

        st.title("Smart Diagnostic System")

        # maintain session variables
        session_state = initSession()

        # set the datetime
        dt_now = datetime.now().strftime("%Y%m%d")

        # This is your Project Root
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # PAGES
        PAGES = [
                "About",
                "Association Rule Mining",
                "Inference",
        ]

        st.sidebar.title("Navigation")
        selection = st.sidebar.radio("Go to", PAGES)

        #About
        if selection.lower()  == "about":
                st.header("About")
                st.markdown(
                        
                        """

                        Smart diagnostic system tool can assist operators to identify the problem and remedy for the given component description. This tool works by building relational association rule from historical data.
                        
                        Association rules are created by thoroughly analyzing data and looking for frequent if/then patterns. Then, depending on the following two parameters, the important relationships are observed:

                        Support: Support indicates how frequently the if/then relationship appears in the database.

                        Confidence: Confidence tells about the number of times these relationships have been found to be true.

                        """
                )


        #Training
        elif selection.lower().replace(" ", "") == "associationrulemining":
                mode = "train"
                st.header("Association Rule Mining (ARM)")

                # drag or upload csv file
                st.set_option('deprecation.showfileUploaderEncoding', False)
                csv_file_buffer = st.file_uploader("Upload the work order file (.csv)", type=["csv"], key="train_upload")

                if csv_file_buffer is not None:
                        dataset,df_replaceNAN = process_csv(csv_file_buffer, mode)
                        st.write(dataset.head())
                        st.write(df_replaceNAN.head())
                
                st.subheader("Generate Rules")

                types_of_algos = st.selectbox ("Select Algorithm", ["Apriori", "FPgrowth"], key="algo_types")

                if st.button("Build rules"):
                        if types_of_algos == "Apriori":
                                session_state.association_rule = runApriori(dataset)
                        elif types_of_algos == "FPgrowth":
                                session_state.association_rule = runFPgrowth(dataset,df_replaceNAN)
                        elif types_of_algos == "FPmax":
                                session_state.association_rule = runFPmax(dataset)

                        if session_state.association_rule is not None:
                                session_state.association_rule['antecedents'] = session_state.association_rule['antecedents'].apply(lambda x: list(x)).astype('unicode')
                                session_state.association_rule['consequents'] = session_state.association_rule['consequents'].apply(lambda x: list(x)).astype('unicode')
                                session_state.association_rule.to_csv('models/association_rule.csv', index=False)
                                session_state.association_rule.to_excel("models/association_rule.xlsx", index=False)

                                                                # save data as hdf5
                                store = pd.HDFStore('models/association_rule.h5')
                                store.put('sd', session_state.association_rule, format='table', data_columns=True)
                                store.close()

                        else:
                                st.error("Association rule is not generated")
                        
                        st.markdown(get_table_download_link(pd.read_csv('./models/association_rule.csv')), unsafe_allow_html=True)
 

                
        #Prediction
        elif selection.lower().replace(" ", "") == "inference":

                mode = "predict"

                st.header("Inference")


                pInput = st.text_area(
                                label = "Input codes",
                                key = "codes")

                pType = 'consequents'#st.selectbox(label = "Association you want to determine", options = ['antecedents', 'consequents'], key="pType")

                # codes = frozenset([str(s) for s in pInput.split(',')])

                metrics = st.checkbox("Additional options", key="metrics")
                
                if metrics:
                        support, confidence, lift = process_metrics()
                else:
                        support, confidence, lift = 0.01,0.01,0.01

                if st.button("Submit"):
                        
                        # load data as hdf5
                        # association_rules = pd.read_hdf('models/association_rule.h5')
                        association_rules = pd.read_csv('models/association_rule.csv')
                        association_rules = association_rules.reset_index(drop=True)                        
                        
                        for i in "'[]":
                                pInput = pInput.replace(i,"")                        
                        pInput = pInput.strip()
                        codes_parsed = [pInput.lower()]
                        # codes_parsed = [str(s) for s in pInput.split(',')]
                        # codes_parsed = ['Replace', 'NA']
                        st.info('**CMD code**: ' + pInput)
                        codes = pd.Series([codes_parsed] * len(association_rules), index = association_rules.index)
                        codes = codes.astype(str)
                    
                        # st.dataframe(association_rules.head(5))
                        # st.info(association_rules[association_rules['consequents'] == codes])

                        if pType == "antecedents":
                                df = association_rules[(association_rules['consequents'] == codes) & (association_rules['support'] > support) & (association_rules['confidence'] > confidence) & (association_rules['lift'] > lift)]
                                # df['consequents'] = association_rules['consequents'].apply(lambda x: list(x)).astype('unicode')
                                # df['antecedents'] = association_rules['antecedents'].apply(lambda x: list(x)).astype('unicode')
                                st.dataframe(df[['antecedents', 'support', 'confidence', 'lift']].reset_index(drop=True))

                        elif pType == "consequents":
                                df = association_rules[ (association_rules['antecedents']== codes) & (association_rules['support'] > support) & (association_rules['confidence'] > confidence) & (association_rules['lift'] > lift)]
                                # df['consequents'] = association_rules['consequents'].apply(lambda x: list(x)).astype('unicode')
                                # df['antecedents'] = association_rules['antecedents'].apply(lambda x: list(x)).astype('unicode')
                                df_to_display = df.copy()
                                df_to_display.rename(columns={'consequents':'PCR Codes'},inplace=True)
                                if len(df_to_display):  
                                        st.table(df_to_display[['PCR Codes', 'support', 'confidence', 'lift']].reset_index(drop=True))
                                else:
                                        st.warning(f'No associated PCR code found for support = \
                                                {support}, confidence = {confidence}, lift = {lift}')

        st.sidebar.markdown("#### **Copyright &copy; 2022 DA REAMS, Siemens Mobility**")
        # st.sidebar.info(
        #         """
        #         This app is maintained by Vinod. You can reach for support at
        #         [vinod.rajendran@siemens.com](vinod.rajendran@siemens.com).
        # """
        # )




