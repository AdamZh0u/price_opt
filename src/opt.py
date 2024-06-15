from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from typing import List

app = FastAPI()

class ProductParams(BaseModel):
    model: str
    purchase_cost: float
    shipping_and_tax: float
    latest_amazon_delivery_fee_usd: float
    vat_local: float
    damage_rate: float
    promotion_discount: float
    ppc_cost: float
    exchange_rate: float
    price_lower_bound: float
    price_upper_bound: float
    sales_at_lower_bound: int
    sales_at_upper_bound: int

class OptimizeResponse(BaseModel):
    optimized_price: List[float]
    expected_sales: List[int]
    fba_commission_local: List[float]
    labor_cost_6_percent: List[float]
    storage_cost_2_percent: List[float]
    net_income_rmb: List[float]
    gross_profit_rmb: List[float]
    max_profit: float

def preprocess_file(df_params):
    df_params['slope'] = (df_params['sales_at_upper_bound'] - df_params['sales_at_lower_bound']) / (df_params['price_upper_bound'] - df_params['price_lower_bound'])
    df_params['intercept'] = df_params['sales_at_lower_bound'] - df_params['slope'] * df_params['price_lower_bound']
    return df_params

def optimize(df_params):
    n_product = len(df_params)
    model = pyo.ConcreteModel()

    def price_bounds_rule(model, i):
        return df_params.loc[i, 'price_lower_bound'], df_params.loc[i, 'price_upper_bound']

    model.prices = pyo.Var(range(n_product), domain=pyo.NonNegativeReals, bounds=price_bounds_rule)

    def total_profit_rule(model):
        a = df_params['slope']
        b = df_params['intercept']

        fba_commission = [model.prices[i] * 0.15 for i in range(n_product)]
        manual_fee = [model.prices[i] * 0.06 for i in range(n_product)]
        storage_fee = [model.prices[i] * 0.02 for i in range(n_product)]
        product_income = [model.prices[i] * (1 - df_params['damage_rate'][i]) - df_params['latest_amazon_delivery_fee_usd'][i] -
                            fba_commission[i] - df_params['vat_local'][i] - df_params['promotion_discount'][i] -
                            df_params['ppc_cost'][i] - manual_fee[i] - storage_fee[i] for i in range(n_product)]
        product_profit_rmb = [product_income[i] * df_params['exchange_rate'][i] - df_params['purchase_cost'][i] -
                                df_params['shipping_and_tax'][i] for i in range(n_product)]

        profit = sum(product_profit_rmb[i] * (a[i] * model.prices[i] + b[i]) for i in range(n_product))
        return profit

    model.total_profit = pyo.Objective(rule=total_profit_rule, sense=pyo.maximize)

    solver = pyo.SolverFactory('ipopt')
    result = solver.solve(model)

    optimized_prices = [pyo.value(model.prices[i]) for i in range(n_product)]
    max_profit = pyo.value(model.total_profit)
    return optimized_prices, max_profit

def result_process(df_params, optimized_prices):
    opt_price = np.array(optimized_prices).round(2)
    df_result = df_params.copy()
    df_result['optimized_price'] = opt_price
    df_result['expected_sales'] = (df_result['slope'] * df_result['optimized_price'] + df_result['intercept']).round(0).astype(int)
    df_result['fba_commission_local'] = df_result['optimized_price']*0.15
    df_result['labor_cost_6_percent'] = df_result['optimized_price']*0.06
    df_result['storage_cost_2_percent'] = df_result['optimized_price']*0.02
    df_result['net_income_rmb'] = (df_result['optimized_price'] * (1 - df_result['damage_rate']) - df_result['latest_amazon_delivery_fee_usd'] - df_result['fba_commission_local'] - df_result['vat_local'] - df_result['promotion_discount'] - df_result['ppc_cost'] - df_result['labor_cost_6_percent'] - df_result['storage_cost_2_percent']).round(2)

    df_result['gross_profit_rmb'] = (df_result['net_income_rmb'] * df_result['exchange_rate'] - df_result['purchase_cost'] - df_result['shipping_and_tax']).round(2)
    df_result = df_result.drop(columns=['slope','intercept'])
    return df_result

@app.post("/optimize", response_model=OptimizeResponse)
def optimize_endpoint(params: List[ProductParams]):
    try:
        df_params = pd.DataFrame([param.dict() for param in params])
        df_params = preprocess_file(df_params)
        optimized_prices, max_profit = optimize(df_params)
        df_result = result_process(df_params, optimized_prices)
        response = OptimizeResponse(
            optimized_price=df_result['optimized_price'].tolist(),
            expected_sales=df_result['expected_sales'].tolist(),
            fba_commission_local=df_result['fba_commission_local'].tolist(),
            labor_cost_6_percent=df_result['labor_cost_6_percent'].tolist(),
            storage_cost_2_percent=df_result['storage_cost_2_percent'].tolist(),
            net_income_rmb=df_result['net_income_rmb'].tolist(),
            gross_profit_rmb=df_result['gross_profit_rmb'].tolist(),
            max_profit=max_profit
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=52565)
