import streamlit as st
import pandas as pd
import numpy as np
import pyomo.environ as pyo

st.set_page_config(page_title="产品价格优化",
                   page_icon="handbag", layout="wide", initial_sidebar_state='collapsed',
                   menu_items={
                       'Get Help': 'https://www.extremelycoolapp.com/help',
                       'Report a bug': "https://www.extremelycoolapp.com/bug",
                       'About': "# Deego Sales Prediction"
                   })


# 设置页面标题
st.title("产品价格优化")

# 上传 df_params 和 df_variables 文件
st.header("步骤 1: 上传文件")
df_params_file = st.file_uploader(" 文件 (XLSX 格式)", type="xlsx")

if df_params_file:
    # 读取文件
    df_params = pd.read_excel(df_params_file)

    # 显示上传的文件并允许用户修改
    st.header("步骤 2: 修改数据")
    st.subheader("df_params 数据")
    df_params = st.data_editor(df_params)

    # 求解优化
    if st.button("求解优化"):
        # 获取产品数量
        n_product = len(df_params)

        df_params['slope'] = (df_params['价格上界时销量'] - df_params['价格下界时销量']) / (df_params['价格上界'] - df_params['价格下界'])
        df_params['intercept'] = df_params['价格下界时销量'] - df_params['slope'] * df_params['价格下界']

        # 创建模型
        model = pyo.ConcreteModel()

        def price_bounds_rule(model, i):
            return df_params.loc[i, '价格下界'], df_params.loc[i, '价格上界']

        # 定义变量
        model.prices = pyo.Var(range(n_product), domain=pyo.NonNegativeReals, bounds=price_bounds_rule)

        # 定义目标函数
        def total_profit_rule(model):
            a = df_params['slope']
            b = df_params['intercept']

            # FBA佣金（本地国)
            fba_commission = [model.prices[i] * 0.15 for i in range(n_product)]
            # 人工6%
            manual_fee = [model.prices[i] * 0.06 for i in range(n_product)]
            # 仓储成本2%
            storage_fee = [model.prices[i] * 0.02 for i in range(n_product)]
            # 实得收入
            product_income = [model.prices[i] * (1 - df_params['折损（退货率）'][i]) - df_params['最新亚马逊配送费（美金）'][i] -
                              fba_commission[i] - df_params['VAT20%\n（本地国)'][i] - df_params['促销折扣费用'][i] -
                              df_params['PPC推广费用'][i] - manual_fee[i] - storage_fee[i] for i in range(n_product)]
            # 毛利润 rmb
            product_profit_rmb = [product_income[i] * df_params['汇率'][i] - df_params['采购成本'][i] -
                                  df_params['运费及税金￥'][i] for i in range(n_product)]

            profit = sum(product_profit_rmb[i] * (a[i] * model.prices[i] + b[i]) for i in range(n_product))
            return profit

        model.total_profit = pyo.Objective(rule=total_profit_rule, sense=pyo.maximize)

        # 求解模型
        solver = pyo.SolverFactory('ipopt')
        result = solver.solve(model)

        # 输出优化结果
        optimized_prices = [pyo.value(model.prices[i]) for i in range(n_product)]
        max_profit = pyo.value(model.total_profit)

        # 显示优化结果
        st.header("步骤 3: 优化结果")
        st.subheader("优化后的价格")
        opt_price = np.array(optimized_prices).round(2)
        df_result = df_params.copy()
        df_result['优化价格'] = opt_price
        df_result['预期销量'] = (df_result['slope']  * df_result['优化价格'] + df_result['intercept']).round(0).astype(int)
        df_result['FBA佣金（本地国)'] = df_result['优化价格']*0.15
        df_result['人工6%'] = df_result['优化价格']*0.06
        df_result['仓储成本2%'] = df_result['优化价格']*0.02
        df_result['得到人民币'] = (df_result['优化价格'] * (1 - df_result['折损（退货率）']) - df_result['最新亚马逊配送费（美金）'] -  df_result['FBA佣金（本地国)'] - df_result['VAT20%\n（本地国)'] - df_result['促销折扣费用'] - df_result['PPC推广费用'] - df_result['人工6%'] - df_result['仓储成本2%']).round(2)

        df_result['毛利润 rmb'] = (df_result['得到人民币'] * df_result['汇率'] - df_result['采购成本'] - df_result['运费及税金￥']).round(2)
        df_result = df_result.drop(columns=['slope','intercept'])
        st.data_editor(df_result)

        st.subheader("最大利润")
        st.write(f"{max_profit:.3f}")
else:
    st.warning("请上传xlsx 文件")