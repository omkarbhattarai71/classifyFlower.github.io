import streamlit as st
import pickle
from PIL import Image
lin_model = pickle.load(open('lin_model.pkl', 'rb'))
log_model = pickle.load(open('log_model.pkl', 'rb'))
svc_model = pickle.load(open('svc_model.pkl', 'rb'))
dtree_model = pickle.load(open('dtree_model.pkl', 'rb'))

# image1 = Image.open('C:/Users/omkar.LAPTOP-S9C2IDIL/PycharmProjects/streamlit/setosa.jpg')
image1 = Image.open('setosa.jpg')
image2 = Image.open('virginica.jpg')
image3 = Image.open('versicolor.jpg')


def classify(num):
    if num<0.5:
        return 'Setosa'
    elif num<1.5:
        return 'Versicolor'
    else:
        return 'Virginica'

# def classify(num):
#     if num<0.5:
#         return s
#     elif num<1.5:
#         return ve
#     else:
#         return vi

# Front End
def main():
    st.title("Web Based Machine Learning Model")
    html_temp = """
    <div style= "background-color:blue; padding:10px">
    <h2 style = "color:white;text-align:center;"> Classification of Iris Flower</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['Linear Regression', 'Logistic Regression', 'SVM','Decision Tree']
    option = st.sidebar.selectbox('Please Select the Model you would like to use', activities)
    st.subheader(option)
    st.spinner("Hello")
    sl = st.slider('Select Sepal Length', 0.0,12.0)
    sw = st.slider('Select Sepal Width', 0.0,12.0)
    pl = st.slider('Select Petal Length', 0.0,12.0)
    pw = st.slider('Select Petal Width', 0.0,12.0)
    inputs = [[sl, sw, pl, pw]]
    if st.button('Classify'):
        if option=='Linear Regression':
            st.success(classify((lin_model.predict(inputs))))
        elif option=='Logistic Regression':
            st.success(classify((log_model.predict(inputs))))
        elif option=='SVM':
            st.success(classify((svc_model.predict(inputs))))
        else:
            st.success(classify((dtree_model.predict(inputs))))
if __name__=="__main__":
    main()

# s = st.image(image1, caption='Setosa Flower')
# vi = st.image(image2, caption='Virginica Flower')
# ve = st.image(image3, caption='Versicolor Flower')