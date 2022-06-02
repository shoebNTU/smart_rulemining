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

import sys


# import visualization tools 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
# import seaborn as sns 

def process_csv(csv_file_buffer, mode):
        df = pd.read_csv(csv_file_buffer)
        # Creating a text element to let the reader know that the data is loading.
        data_load_state = st.text('Loading data...')
        df_replaceNAN = df.replace(np.nan, 'NA', regex=True)
        processed_dataset = df_replaceNAN[['ATS', 'CMD', 'Problem Code', 'Cause Code', 'Remedy Code']].to_numpy()
        te = TransactionEncoder()
        te_ary = te.fit(processed_dataset).transform(processed_dataset)
        transformed_dataset = pd.DataFrame(te_ary, columns=te.columns_)

        # Notifying the reader that the data was successfully loaded.
        data_load_state.text('Loading data...done!')

        return transformed_dataset

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

def runFPgrowth(dataset):
        start = datetime.now()
        frequent_itemsets_fp = fpgrowth(dataset,min_support=0.1 ,use_colnames=True) 
        deltatime = datetime.now() - start
        frequent_itemsets_fp.sort_values(by='support',ascending=False,inplace=True)
        association_rule_fp = association_rules(frequent_itemsets_fp,metric='confidence',min_threshold=0.1) 
        association_rule_fp.sort_values(by='leverage',ascending=False,inplace=True)    
        association_rule_fp['length'] = association_rule_fp['consequents'].apply(lambda x: len(x))
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
        support = st.sidebar.slider("support", 0.0, 1.0, 0.1)
        confidence = st.sidebar.slider("confidence", 0.0, 1.0,0.1)
        lift = st.sidebar.slider("lift", 0.0, 1000.0,0.1)
        length = st.sidebar.slider("length", 1, 10,3)

        return support, confidence, lift, length


def initSession():

        sessionstate_init = SessionState.get(

                association_rules = None,
                support	= 0.1,
                confidence = 0.1,
                lift = 0.1,
                length = 1	

        )

        return sessionstate_init


if __name__ == "__main__":

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
                        dataset = process_csv(csv_file_buffer, mode)

                
                st.subheader("Generate Rules")

                types_of_algos = st.selectbox ("Select Algorithm", ["Apriori", "FPgrowth"], key="algo_types")

                if st.button("Build rules"):
                        if types_of_algos == "Apriori":
                                session_state.association_rule = runApriori(dataset)
                        elif types_of_algos == "FPgrowth":
                                session_state.association_rule = runFPgrowth(dataset)
                        elif types_of_algos == "FPmax":
                                session_state.association_rule = runFPmax(dataset)

                        if session_state.association_rule is not None:
                                # association_rule['antecedents'] = association_rule['antecedents'].apply(lambda x: list(x)).astype('unicode')
                                # association_rule['consequents'] = association_rule['consequents'].apply(lambda x: list(x)).astype('unicode')
                                session_state.association_rule.to_csv('models/association_rule.csv', index=False)
                                session_state.association_rule.to_excel("models/association_rule.xlsx", index=False)  
                        else:
                                st.error("Association rule is not generated")

                
        #Prediction
        elif selection.lower().replace(" ", "") == "inference":

                mode = "predict"

                st.header("Inference")


                pInput = st.text_area(
                                label = "Input codes",
                                key = "codes")

                pType = st.selectbox(label = "Association you want to determine", options = ['antecedents', 'consequents'], key="pType")

                # codes = frozenset([str(s) for s in pInput.split(',')])

                codes = frozenset(s.strip() for s in pInput.split(','))

                metrics = st.checkbox("Additional options", key="metrics")

                if metrics:
                        support, confidence, lift, length = process_metrics()
                else:
                        support, confidence, lift, length = session_state.support, session_state.confidence, session_state.lift, session_state.length

                if st.button("Submit"):
                        association_rules = session_state.association_rule
                        if pType == "antecedents":
                                df = association_rules[(association_rules['consequents'] == codes) & (association_rules['support'] >= support) & (association_rules['confidence'] >= confidence) & (association_rules['lift'] >= lift) & (association_rules['length'] >= length)]
                                df['consequents'] = association_rules['consequents'].apply(lambda x: list(x)).astype('unicode')
                                df['antecedents'] = association_rules['antecedents'].apply(lambda x: list(x)).astype('unicode')
                                st.dataframe(df[['antecedents', 'support', 'confidence', 'lift']])

                        elif pType == "consequents":
                                df = association_rules[(association_rules['antecedents']== codes) & (association_rules['support'] >= support) & (association_rules['confidence'] >= confidence) & (association_rules['lift'] >= lift) & (association_rules['length'] >= length)]
                                df['consequents'] = association_rules['consequents'].apply(lambda x: list(x)).astype('unicode')
                                df['antecedents'] = association_rules['antecedents'].apply(lambda x: list(x)).astype('unicode')
                                st.dataframe(df[['consequents', 'support', 'confidence', 'lift']])


        st.sidebar.markdown("#### **Copyright &copy; 2020 DA REAMS, Siemens Mobility**")
        st.sidebar.info(
                """
                This app is maintained by Vinod. You can reach for support at
                [vinod.rajendran@siemens.com](vinod.rajendran@siemens.com).
        """
        )




