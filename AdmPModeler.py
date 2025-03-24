from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain

# Initialize the language model
llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0,
    openai_api_key="INSERT YOUR OPENAI KEY"
)

# STEP 1 - Law Article -> Initial Flowchart
system_msg_1 =  """**Contesto**: Sei un esperto nell’analisi di procedure amministrative. 
**Obiettivo**: Crei rappresentazioni strutturate della sequenza di attivita che compongono una procedura amministrativa.
**Input**: <FINALITA DELLA PROCEDURA> e <ARTICOLO DI LEGGE> che descrive la procedura amministrativa.
**Output**: <FLOWCHART> della procedura.
**Stile**: Segui un approccio STEP-BY-STEP:
    - INIZIO;
    - Analizzi attentamente la <FINALITA DELLA PROCEDURA>;
    - Segui attentamente l’ordine logico dell’<ARTICOLO DI LEGGE>;
    - Identifica e suddividi le attivita della procedura in passaggi ben definiti;
    - Modella le sequenze e i gateway OR, AND o XOR che collegano le attivita;
    - FINE.
**Vincoli**: 
    - Aggiungi al <FLOWCHART> solo le attivita che sono esplicitamente descritte nell'<ARTICOLO DI LEGGE>;
    - Fai il <FLOWCHART> corto se l'<ARTICOLO DI LEGGE> è corto.

**Audience**: Il destinatario è un analista tecnico di procedure amministrative, quindi il flowchart deve essere perfetto.
**Risposta**: Restituisci SOLO il <FLOWCHART> in formato testuale, con indentazioni e simboli:
    - Usa (↓) per indicare il passaggio all'attivita successiva;
    - Usa simboli di ramificazione (├── e └──) e la formula "Se [condizione] → [azione]"  per rappresentare i gateway;
    - Scrivi (facoltativa) per specificare se un'attivita è facoltativa."""
human_msg_1 = """##INPUT
<FINALITA DELLA PROCEDURA>:
{purpose}
<ARTICOLO DI LEGGE>:
{law_article}

##OUTPUT
<FLOWCHART>:"""

prompt_1 = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_msg_1),
    HumanMessagePromptTemplate.from_template(human_msg_1)
])

chain1 = LLMChain(llm=llm, prompt=prompt_1, output_key="flowchart")

# STEP 2 - Law Article + Initial Flowchart -> Updated Flowchart
system_msg_2 = """**Contesto**: Sei un esperto nel controllo e correzione di procedure amministrative.
**Obiettivo**: Controlli e, se necessario correggi, il flowchart che descrive una procedura amministrativa.
**Input**:
    - <ARTICOLO DI LEGGE>: La fonte originale che descrive la procedura amministrativa.
    - <FLOWCHART>: Il flowchart della procedura amministrativa, estratto dall'<ARTICOLO DI LEGGE>.
**Output**: <FLOWCHART DEFINITIVO>: Il nuovo <FLOWCHART> controllato e corretto.
**Stile**: Segui un approccio STEP-BY-STEP:
    - Analizzi attentamente l'<ARTICOLO DI LEGGE> e il <FLOWCHART>;
    - Per ogni Attività del <FLOWCHART>:
    - Aggiungi o modifichi l'identificativo univico dell'attività utilizzando i commi dell'<ARTICOLO DI LEGGE> come riferimento;
    - Confronti l'<ARTICOLO DI LEGGE> e il <FLOWCHART>;
    - Se il <FLOWCHART> presenta delle differenze con la descrizione della procedura nell'<ARTICOLO DI LEGGE>, modifichi il <FLOWCHART>;
    - Aggiungi al <FLOWCHART> le attività mancanti rispetto all'<ARTICOLO DI LEGGE>;
    - Togli dal <FLOWCHART> le attività di troppo rispetto all'<ARTICOLO DI LEGGE>.
**Audience**: Il destinatario è un analista tecnico di procedure amministrative, quindi il <FLOWCHART DEFINITIVO> deve essere preciso, corretto e coerente con l'<ARTICOLO DI LEGGE>.
**Risposta**: Restituisci solo il <FLOWCHART DEFINITIVO> in formato testuale con indentazioni e simboli."""
human_msg_2 = """##INPUT
<ARTICOLO DI LEGGE>:
{law_article}
<FLOWCHART>:
{flowchart}
##OUTPUT
<FLOWCHART DEFINITIVO>:"""

prompt_2 = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_msg_2),
    HumanMessagePromptTemplate.from_template(human_msg_2)
])

chain2 = LLMChain(llm=llm, prompt=prompt_2, output_key="updated_flowchart")

# STEP 3 - Flowchart -> Intermediate CSV File
system_msg_3 = """**Contesto**: Sei un esperto nella trasformazione di procedure amministrative.
**Obiettivo**: Trasformi un flowchart che descrive una procedura in una <TABELLA>.
**Input**: <FLOWCHART> della procedura amministrativa che contiene Attivita e Sotto-Attivita.
**Output**: <TABELLA> che contiene 3 colonne:
    1. Nome Attivita (breve);
    2. Identificativo Attivita;
    3. Identificativi delle Attivita Precedenti.
**Stile**: Segui un approccio STEP-BY-STEP:
    - Per ogni Attivita e Sotto-Attivita del <FLOWCHART>:
        - Crei una riga nella <TABELLA> aggiungendo i due campi ''Nome Attivita'' e un ''Identificativo Attivita'' (l' identificativo deve essere univoco);
    - Per ogni coppia ''Nome Attivita'' e ''Identificativo Attivita'' nella <TABELLA> che hai creato:
        - Determini gli ''Identificativi delle Attivita Precedenti'' utilizzando il <FLOWCHART> per le precedenze e la <TABELLA> per gli identificativi;
        - Aggiungi il terzo campo ''Identificativi delle Attivita Precedenti''; 
**Audience**: Il destinatario è un analista tecnico di procedure amministrative, quindi la tabella deve essere perfetta.
**Risposta**: Restituisci solo la <TABELLA> in formato csv."""

human_msg_3 = """##INPUT
<FLOWCHART>:
{flowchart}
##OUTPUT
<TABELLA>:"""

prompt_3 = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_msg_3),
    HumanMessagePromptTemplate.from_template(human_msg_3)
])

chain3 = LLMChain(llm=llm, prompt=prompt_3, output_key="intermediate_csv")

# STEP 4 - Law Article + Intermediate CSV -> Final CSV
system_msg_4 = """**Contesto**: Sei un data entry operator di tabelle che rappresentano procedure amministrative.
**Obiettivo**: Inserisci i valori necessari per completare una tabella che descrive una procedura amministrativa.
**Input**: <ARTICOLO DI LEGGE>, <TABELLA ATTIVITA DELLA PROCEDURA>
**Output**: <TABELLA DEI METADATI> (si ottiene completando <TABELLA ATTIVITA DELLA PROCEDURA>) che contiene:
    1. Nome dell'Attivita (gia presente in <TABELLA ATTIVITA DELLA PROCEDURA>);
    2. Identificativo dell'Attivita (gia presente in <TABELLA ATTIVITA DELLA PROCEDURA>);
    3. Descrizione (breve descrizione dell'attivita);
    4. Riferimento normativo puntuale (riferimento normativo più preciso possibile);
    5. Attore (entita che esegue l'attivita);
    6. Destinatario (entita che riceve l'effetto dell'attivita);
    7. Documenti (documenti coinvolti nell'attivita);
    8. Riferimento temporale (durata stimata per completare l'attivita o limite massimo);
    9. Output (risultato o la richiesta o le decisioni finali dell’attivita);
    10. Moduli (moduli utilizzati durante l'attivita).

**Stile**: Segui un approccio STEP-BY-STEP:
    - Per ogni riga della <TABELLA ATTIVITA DELLA PROCEDURA>:
        - Crei una nuova riga in <TABELLA DEI METADATI> e inserisci Nome dell'Attivita e Identificativo dell'Attivita;
        - Analizzi attentamente l'<ARTICOLO DI LEGGE>;
        Per ogni metadato richiesto:
            - Cerchi in <ARTICOLO DI LEGGE> il metadato dell'attivita;
            - Se è presente:
                - Inserisci il metadato nella colonna corrispondente;
            - Altrimenti: Inserisci 'N/A'.
**Audience**: Il destinatario è uno specialista in gestione documentale di procedure amministrative, quindi la tabella deve essere perfetta.
**Risposta**: Restituisci solo la <TABELLA DEI METADATI> in formato csv."""
human_msg_4 = """##INPUT
<ARTICOLO DI LEGGE>:
{law_article}
<TABELLA ATTIVITA DELLA PROCEDURA>:
{csv_intermedio}
##OUTPUT
<TABELLA DEI METADATI>:"""

prompt_4 = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_msg_4),
    HumanMessagePromptTemplate.from_template(human_msg_4)
])

chain4 = LLMChain(llm=llm, prompt=prompt_4, output_key="final_csv")


# Run the chain step-by-step
res1 = chain1.run(law_article=law_article, purpose=purpose)
res2 = chain2.run(law_article=law_article, flowchart=res1)
res3 = chain3.run(flowchart=res2)
res4 = chain4.run(articolo=law_article, csv_intermediate=res3)

# Print results
print("Initial Flowchart:\n", res1)
print("Updated Flowchart:\n", res2)
print("Intermediate CSV:\n", res3)
print("Final CSV:\n", res4)


# Saving tables in txt and csv formats
name_intermediate_table_in_txt = "intermediate_table_in_txt"
name_intermediate_table_in_csv = "intermediate_table_in_csv"
name_table_in_txt = "table_in_txt"
name_table_in_csv = "table_in_csv"

with open(f"{name_intermediate_table_in_txt}.txt", "w", encoding="utf-8") as txt_file:
    txt_file.write(csv_intermediate)

with open(f"{name_intermediate_table_in_csv}.csv", "w", encoding="utf-8") as csv_file:
    csv_file.write(csv_intermediate)

    
with open(f"{name_table_in_txt}.txt", "w", encoding="utf-8") as txt_file:
    txt_file.write(res4)

with open(f"{name_table_in_csv}.csv", "w", encoding="utf-8") as csv_file:
    csv_file.write(res4)