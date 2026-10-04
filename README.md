# 🏷️ LabelWise: TabPFN ROI Calculator

**Stop overpaying for data labels.** LabelWise is an interactive tool that calculates the exact Return on Investment (ROI) for your data labeling budget using the instant, zero-training capabilities of **TabPFN-3.5**.

## 🌟 The Problem
Data science teams constantly struggle with the question: *"If we buy 10,000 more labels, how much will our model actually improve, and is it worth the $5,000 cost?"* 
Generating a scaling curve to answer this traditionally requires retraining models at every data size, taking hours or days.

## 💡 The Solution
LabelWise leverages **TabPFN-3.5's** foundation model architecture to evaluate scaling curves in **seconds**. It translates technical performance metrics into business metrics (Cost, Time, and Marginal ROI), automatically identifying the "knee" of the curve where diminishing returns begin.

## 🚀 Key Features
- **Zero-Training Scaling Curves**: Plot performance vs. data size instantly.
- **Business Translation**: Input your labeler's hourly rate and speed to see exact dollar costs.
- **Marginal ROI Analysis**: Visualizes exactly where the next dollar stops yielding meaningful accuracy gains.
- **Automated Stopping Point**: Algorithmically recommends the optimal number of labels to buy.

## 🛠️ Tech Stack
- **Backend**: Python, TabPFN-3.5, Scikit-Learn
- **Frontend**: Streamlit
- **Visualization**: Plotly

## 🏃 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/labelwise-tabpfn.git
   cd labelwise-tabpfn```
   
2. Install dependencies:
 ```bash
   pip install -r requirements.txt ```

3. Run the app:
 ```bash
   streamlit run app.py```
