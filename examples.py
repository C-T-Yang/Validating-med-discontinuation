# Add PRN, Taking, StopTaking flags to the dataset

import pandas
import csv
from glob import glob
from pathlib import Path
import pandas as pd
from collections import Counter
from pprint import pprint as pp 
from tqdm import tqdm
import os
import numpy as np
from nltk import tokenize
import regex as rege
from nltk.tokenize import word_tokenize, TreebankWordTokenizer as twt





stopPassiveVerbs = pd.read_csv('path_to_file')
takePassiveVerbs = pd.read_csv('path_to_file')
takeActiveVerbs = pd.read_csv('path_to_file')
stopActiveVerbs = pd.read_csv('path_to_file')
prn_verbs = pd.read_csv('path_to_file')

def toLower(ls):
    return [i.lower() for i in ls]
stopPassiveVerbs = set(toLower(stopPassiveVerbs))
stopActiveVerbs = set(toLower(stopActiveVerbs))
takePassiveVerbs = set(toLower(takePassiveVerbs))
takeActiveVerbs = set(toLower(takeActiveVerbs))
prn_verbs = set(toLower(prn_verbs))

prnPattern = [f'({m})' for m in list(prn_verbs)]
prnPattern = '|'.join(prnPattern)

bullets = ['*', '-', '•','.','?']
bulletsPattern = [f'(\{m})' for m in bullets]
bulletsPattern = '|'.join(bulletsPattern)

medication_names = load_medication('path_to_file')
data_tables = load_data_file('path_to_file')

# example
# TypicalAntipsychotic = [
#         ['Haloperidol', 'Haldol Decanoate', 'Haldol']
#         ]



medicationNM = sum(medication_names,[])
medicationNM = set(toLower(medicationNM))
medicationNMPattern = [f'({m})' for m in medicationNM]
medicationNMPattern = '|'.join(medicationNMPattern)


def combineActiveReg(group, active_verbs, num):
    group = [f'({i})' for i in group]
    group = '|'.join(group)
    active_verbs = [fr'\b({i})\b' for i in active_verbs]
    active_verbs = '|'.join(active_verbs)
    return rf'({active_verbs})([^-.]){{0,{num}}}({group})'

def combinePassiveReg(group, passive_verbs, num):
    group = [f'({i})' for i in group]
    group = '|'.join(group)
    active_verbs = [fr'\b({i})\b' for i in passive_verbs]
    active_verbs = '|'.join(active_verbs)
    return rf'({group})([^-.]){{0,{num}}}({active_verbs})'


def checkForVerb(verb): #passive with buffer
    group = [fr'\b({i})\b' for i in verb]
    group = '|'.join(group)
    return rf'({group})'

def checkForVerbReverse(verb): #passive with buffer
    group = [fr'\b({i[::-1]})\b' for i in verb]
    group = '|'.join(group)
    return rf'({group})'


def createString(x):
    #i, tokens = list(zip(*x))
    return " ".join(x)



for table in data_tables:
    df_chunks = pd.read_csv(table, chunksize=100000, delimiter='|', on_bad_lines='skip', dtype = str)
    med_group = Path(table).name.split('_')[3]
    

    output_file =  open('path_to_file', 'w')
    field_names = load_header('path_to_file')
    output_file.write(field_names)

    selTakeCounts = Counter()
    selStopCounts = Counter()
    selPriorCounts = Counter()

    for chunk in tqdm(df_chunks):
        chunk = chunk.replace({np.nan:''})
  
        for row in chunk.itertuples():
            selectedPRNDict = {}
            selectedTaking = []
            selectedStopTaking = []
            full_notes = str(row.NoteTXT).lower()
            notes = full_notes
            fullSpan = twt().span_tokenize(notes)
            

            stopMedSearch = rege.search('(stop taking these medications)', notes, rege.IGNORECASE)
            startMedSearch = rege.search('(start taking these medications)|(take these medications)', notes, rege.IGNORECASE)
            listEndSearch = list(rege.finditer('phone:', notes, rege.IGNORECASE))
            
            stopMedLs = ''
            startMedLs = ''
            stopMedStart = None
            stopMedEnd = None
            startMedStart = None
            startMedEnd = None
            
            # search for lists of medications within notes which declares starting or stopping 
            if listEndSearch:
                listEndSearch = listEndSearch[-1]
                listED = listEndSearch.start()
                if stopMedSearch and not startMedSearch:
                    stopMedStart = stopMedSearch.start()
                    stopMedEnd = listED
                    stopMedLs = notes[stopMedStart: listED]
                    notes = notes[:stopMedSearch.start()] + '[LISTPLACEHOLDER]' + notes[listED:]
                elif startMedSearch and not stopMedSearch:
                    startMedStart = startMedSearch.start()
                    startMedEnd = listED
                    startMedLs = notes[startMedStart: listED]
                    notes = notes[:startMedStart] + '[LISTPLACEHOLDER]' + notes[listED:]
                elif stopMedSearch and startMedSearch:
                    stopMedStart = stopMedSearch.start()
                    stopMedEnd = startMedSearch.start()
                    startMedStart = startMedSearch.start()
                    startMedEnd = listED
                    stopMedLs = notes[stopMedStart: startMedStart]
                    startMedLs = notes[startMedStart: listED]
                    notes = notes[:stopMedStart] + '[LISTPLACEHOLDER]' + notes[listED:]

            
            # append to corresponding list if stop or start is found 
            found = rege.search(medicationNMPattern, stopMedLs, rege.IGNORECASE)
            if found:
                result = ((0,0), f'[stop list] {found.group()}', (stopMedStart, stopMedEnd))
                selectedStopTaking.append(result)
                selectedPRNDict[((0,0), f'[stop list] {found.group()}')] = None
            found = rege.search(medicationNMPattern, startMedLs, rege.IGNORECASE)
            if found:
                result = ((0,0), f'[take list] {found.group()}',(startMedStart, startMedEnd))
                selectedTaking.append((result))
                selectedPRNDict[(0,0), f'[take list] {found.group()}'] = None


            # tokenenizes text
            tokens = word_tokenize(notes)
            eTokens = list(zip(range(len(tokens)),tokens,fullSpan))


            # searches within a given span from the key token 
            for i, token, fullSpan in eTokens:

                if token in medicationNM:
                    spacing = 7
                    start = max(0,i-spacing)
                    end = min(len(tokens), i+spacing)

                    preTokens = []
       
                    for a,c,b in reversed(eTokens[start:i+1]):
                        preTokens.append((a,c,b))
                        if rege.search(bulletsPattern, c):
                            break
                    preTokens = list(reversed(preTokens))

                    postTokens = []
                    for a,c,b in eTokens[i: end]:
                        postTokens.append((a,c,b))
                        if rege.search(bulletsPattern, c):
                            break
                    
                    preTokensStr = ''
                    _,b,_ = list(zip(*preTokens))
                    if preTokens:
                        preTokensStr = createString(b)
                    preTokenSearch = rege.search(prnPattern, preTokensStr)
                    
                    postTokensStr = ''
                    _,b,_ = list(zip(*postTokens))
                    if postTokens:
                        postTokensStr = createString(b)
                    postTokenSearch = rege.search(prnPattern, postTokensStr)
                    
                    for j, token, fullSpan  in preTokens:
                        tokenStr = createString(tokens[j:i+1])
                        result = ((j,i+1), tokenStr, fullSpan)
                        
                        if preTokenSearch:
                            selectedPRNDict[((j,i+1), tokenStr)] = preTokenSearch.group()
                        else:
                            selectedPRNDict[((j,i+1), tokenStr)] = None
                        
                        if token in takeActiveVerbs:  
                            selectedTaking.append(result)
                            continue
                        if token in stopActiveVerbs:
                            selectedStopTaking.append(result)
                            continue
                        if token == 'on':
                            nextT = min(len(tokens), j+1)
                            if tokens[nextT] == 'the':
                                nextT = min(len(tokens), j+2)
                            if tokens[nextT] in medicationNM:
                                selectedTaking.append(result)
                            continue
                        
                    for j, token, fullSpan in postTokens:
                        tokenStr = createString(tokens[i: j+1])
                        result = ((i, j+1), tokenStr, fullSpan)
                        if postTokenSearch:
                            selectedPRNDict[((i, j+1), tokenStr)] = postTokenSearch.group()
                        else:
                            selectedPRNDict[((i, j+1), tokenStr)] = None
                        if token in takePassiveVerbs:   
                            selectedTaking.append(result)
                            continue
                        if token in stopPassiveVerbs:
                            selectedStopTaking.append(result)
                            continue
            
            # searches the matches for key term that indicate a stop term                         
            temp = []
            selectedPriorStop = []

            for j,textString, fullSpan in selectedTaking:
                start, end = j
                spacing = 2
                l_start = max(0, start - spacing)
                l_tokens = eTokens[l_start: end]
                hasStop = False
                for k, token, fullSpan in l_tokens:
                    if token in stopActiveVerbs:
                        hasStop = True
                    if hasStop:
                        q = min(k,l_start)
                        tokenStr = createString(tokens[q: end])
                        result = ((q, end),tokenStr, fullSpan)
                        selectedPriorStop.append(result)
                        selectedStopTaking.append(result)
                        selectedPRNDict[((q, end),tokenStr)] = selectedPRNDict[(j,textString)]
                        break
                    
                if not hasStop:
                    temp.append((j,textString, fullSpan))
            selectedTaking = temp


            selectedStopTaking = [elem for elem in selectedStopTaking if 'last time this was given' not in elem[1]]
            selectedTaking = [elem for elem in selectedTaking if 'last time this was given' not in elem[1]]
            selectedPriorStop = [elem for elem in selectedPriorStop if 'last time this was given' not in elem[1]]
            


  
            prnCount = 0
            for tokenSpan, selected, fullSpan in selectedStopTaking:
                if selectedPRNDict[(tokenSpan, selected)]:
                    prnCount+=1 
            
            if prnCount != len(selectedStopTaking):
                selectedStopTaking = [(a,b,c) for a,b,c in selectedStopTaking if not selectedPRNDict[(a,b)]]

            prnCount = 0
            for tokenSpan, selected, fullSpan in selectedTaking:
                if selectedPRNDict[(tokenSpan, selected)]:
                    prnCount+=1 
            
            if prnCount != len(selectedTaking):
                selectedTaking = [(a,b,c) for a,b,c in selectedTaking if not selectedPRNDict[(a,b)]]


            PatientID = str(row.PatientID)
            NoteID = str(row.NoteID)
            ContactDateRealNBR = str(row.ContactDateRealNBR)
            NoteCSNID = str(row.NoteCSNID)
            ContactDTS = str(row.ContactDTS)
            UnifiedClinicalNoteTypeDSC = str(row.UnifiedClinicalNoteTypeDSC)
            NoteTXT = str(row.NoteTXT)
            PRN = "0"
            StopTaking = "0"
            Taking = "0"
            

            start1 = None 
            start2 = None 
            end1 = None
            end2 = None
            lookCount = 25      

            
            stopTokenSpan = None
            stopFullSpan= None
            takeTokenSpan = None
            takeFullSpan = None
            priorTokenSpan = None
            priorFullSpan = None

            selectedPRNStr1 = ''
            selectedPRNStr2 = ''


            # assigns labels for the stop or stop column 
            if selectedStopTaking:
                selectedStopTaking.sort(key=lambda x: x[2][0])
                stopTokenSpan, selectedStopTaking, stopFullSpan = list(zip(*selectedStopTaking))
                
                selectedStopTaking = selectedStopTaking[-1]
                stopTokenSpan = stopTokenSpan[-1]
                stopFullSpan = stopFullSpan[-1]
                StopTaking = "1"
                start1 = max(0, stopTokenSpan[0] - lookCount)
                end1 = min(len(tokens), stopTokenSpan[1] + lookCount)

                selectedPRNStr1 = selectedPRNDict[(stopTokenSpan, selectedStopTaking)]

            if selectedTaking:

                selectedTaking.sort(key=lambda x: x[2][0])
                takeTokenSpan, selectedTaking, takeFullSpan = list(zip(*selectedTaking))
                selectedTaking = selectedTaking[-1]
                takeTokenSpan = takeTokenSpan[-1]
                takeFullSpan = takeFullSpan[-1]
                Taking = "1"
                start2 = max(0, takeTokenSpan[0] - lookCount)
                end2 = min(len(tokens), takeTokenSpan[1] + lookCount)
                selectedPRNStr2 = selectedPRNDict[(takeTokenSpan, selectedTaking)]

            if Taking == '1' and StopTaking == "1":
                if takeFullSpan[0] > stopFullSpan[0]:
                    StopTaking = '0'
                    start1 = start2
                    end1 = end2
                    selectedPRNStr1 = selectedPRNStr2
                else:
                    Taking = '0'
            if selectedPriorStop:
                priorTokenSpan, selectedPriorStop, priorFullSpan = list(zip(*selectedPriorStop))
                selectedPriorStop = selectedPriorStop[-1]
                priorTokenSpan = priorTokenSpan[-1]
            else:
                selectedPriorStop = ''
                
            if not selectedStopTaking:
                start1 = start2
                end1 = end2
                selectedStopTaking = ''
                selectedPRNStr1 = selectedPRNStr2
            if not selectedTaking:
                selectedTaking = ''

            if selectedPRNStr1:
                PRN = '1'
            
            
            shownText = createString(tokens[start1:end1]) if selectedStopTaking or selectedTaking else ''
            fullText = full_notes 
            

            selTakeCounts.update([selectedTaking])
            selStopCounts.update([selectedStopTaking])
            selPriorCounts.update([selectedPriorStop])

            output_file.write(f'{PatientID}|{NoteID}|{ContactDateRealNBR}|{NoteCSNID}|{ContactDTS}|{UnifiedClinicalNoteTypeDSC}|{PRN}|{StopTaking}|{Taking}|{selectedPriorStop}|{selectedPRNStr1}|{selectedStopTaking}|{selectedTaking}|{stopFullSpan}|{takeFullSpan}|{shownText}|{fullText}\n')
    
 
output_file.close()

