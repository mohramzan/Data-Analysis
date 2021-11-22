import gradio as gr
from joblib import load
import pandas as pd



dv , model = load("train_model.joblib")



# creating a predict function to be passed into gradio UI
def predict(age, job, marital, education, default, housing,
            loan, contact, month,day_of_week,campaign,pdays,
            previous,poutcome,emp_var_rate,cons_price_idx,cons_conf_idx):
  
    customer = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default, 
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome,
        'emp_var_rate': emp_var_rate,
        'cons_price_idx': cons_price_idx,
        'cons_conf_idx': cons_conf_idx,
        
    }
    
    print(customer)
    df_transformed = dv.transform([customer])
    prediction = model.predict_proba(df_transformed)[:,1]
    
#     Desposited = prediction >= 0.50
    
#     result = {
#         "deposit_probability": float(prediction),
#         "Deposited": bool(Deposited)
#     } 
    print(f' The probabilty of depositing in the bank is  : {str(prediction)}')
    
    return str(prediction)




age = gr.inputs.Slider(minimum=1,default = 35, maximum=100, step=1,label = 'Age') #default=data['age'].mean()
job = gr.inputs.Dropdown(choices=["Housemaid", "Services","Admin.","Blue-Collar","Technician",
                                 "Retired","Management","Unemployed","Self-Employed","Unknown",
                                  "Entrepreneur","Student"],label = 'Job')
marital = gr.inputs.Dropdown(choices=["Married", "Single","Divorced","Unknown"],label = 'Marital')
education = gr.inputs.Dropdown(choices=["Basic.4y", "High.School","Basic.6y","Basic.9y","Professional.Course",
                                 "Unknown","University.Degree","Illiterate"],label = 'Education')
default = gr.inputs.Radio(["Yes", "No","Unknown"],label = 'Default',type="index")
housing = gr.inputs.Radio(choices=["Yes", "No","Unknown"],label = 'Housing',type="index")
loan = gr.inputs.Radio(["Yes", "No","Unknown"],type="index",label = 'Loan')
contact = gr.inputs.Radio(["Telephone", "Cellular"],type = "index",label = 'Contact')
month = gr.inputs.Dropdown(choices=['Mar', 'Apr','May', 'Jun', 'Jul', 'Aug','Sep','Oct', 'Nov', 'Dec'],label = 'Month')
day_of_week = gr.inputs.Dropdown(choices=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],label = 'Day of Week')
campaign = gr.inputs.Slider(minimum=1,default = 2, maximum=56, step = 1,label = 'Campaign') 
pdays = gr.inputs.Slider(minimum=0,default = 0, maximum=27, step = 1,label = 'Last Contact(in days)') 
previous = gr.inputs.Slider(minimum=0,default = 0, maximum=7, step = 1,label = 'Previous Contacts') 
poutcome = gr.inputs.Radio(["Nonexistent", "Failure","Success"],label = 'Previous Outcome',type="index")
emp_var_rate = gr.inputs.Slider(minimum=-3,default = 1, maximum=1,step= 1, label = 'Employment Variation Rate')
cons_price_idx = gr.inputs.Slider(minimum=92,default = 94, maximum=95,step = 1, label = 'Consumer Price Index ')
cons_conf_idx = gr.inputs.Slider(minimum=-51,default = -42, maximum=-27, step = 1, label = 'Consumer Confidence Index')




iface = gr.Interface(predict,[age, job, marital, education, default, housing,
            loan, contact, month,day_of_week,campaign,pdays,
            previous,poutcome,emp_var_rate,cons_price_idx,cons_conf_idx],
             outputs = "number",
            interpretation="default",verbose = True
             )
iface.launch(share=True)





