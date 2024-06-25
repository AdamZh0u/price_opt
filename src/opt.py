
import numpy as np
import pyomo.environ as pyo
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from typing import List

app = FastAPI()


class ProductParams(BaseModel):
    model: str  # 型号
    purchase_cost: float  # 采购成本
    shipping_and_tax: float  # 运费及税金
    latest_amazon_delivery_fee_usd: float  # 最新亚马逊配送费（美元）
    vat_local: float  # vat 20% 本地国
    damage_rate: float  # 折损（退货率）
    promotion_discount: float  # 促销折扣费用
    ppc_cost: float  # PPC推广费用
    exchange_rate: float  # 汇率
    price_lower_bound: float  # 价格下界
    price_upper_bound: float  # 价格上界
    sales_at_lower_bound: int  # 价格下界时销量
    sales_at_upper_bound: int  # 价格上界时销量
    deal_sales_lower_bound: float  # 促销最低价格时销量 最低价格=price_lower_bound*0.85
    deal_sales_upper_bound: float  # 促销最高价格时销量 最高价格=price_upper_bound*0.85


class OptimizeResponse(BaseModel):
    optimized_price: List[float]  # 最优价格
    is_deal: List[int]  # 是否做deal
    month_profit_rmb: List[float]  # 月利润
    ld: int  # LD天数
    bd: int  # BD数 一个BD等于7天


def preprocess_file(df_params):
    df_params['slope'] = (df_params['sales_at_upper_bound'] - df_params['sales_at_lower_bound']
                          ) / (df_params['price_upper_bound'] - df_params['price_lower_bound'])
    df_params['intercept'] = df_params['sales_at_lower_bound'] - \
        df_params['slope'] * df_params['price_lower_bound']
    df_params['deal_slope'] = (df_params['deal_sales_upper_bound'] - df_params['deal_sales_lower_bound']
                               ) / (df_params['price_upper_bound'] - df_params['price_lower_bound'])
    df_params['deal_intercept'] = df_params['deal_sales_lower_bound'] - \
        df_params['deal_slope'] * df_params['price_lower_bound']
    return df_params


def optimize(df_params):
    n_product = len(df_params)
    model = pyo.ConcreteModel()

    def price_bounds_rule(model, i):
        return df_params.loc[i, 'price_lower_bound'], df_params.loc[i, 'price_upper_bound']

    model.prices = pyo.Var(
        range(n_product), domain=pyo.NonNegativeReals, bounds=price_bounds_rule)

    # 每个产品是否做deal
    model.deal = pyo.Var(range(n_product), domain=pyo.Binary, bounds=(0, 1))

    # set product 1 deal to 1
    model.deal[0].fix(1)

    # 整个产品deal的天数
    model.A = pyo.Set(initialize=['LD', 'BD'])
    lb = {'LD': 0, 'BD': 0}
    ub = {'LD': 4, 'BD': 1}

    def fb(model, i):
        return (lb[i], ub[i])

    model.deal_days = pyo.Var(
        model.A, bounds=fb, domain=pyo.NonNegativeIntegers)

    # 总利润
    def total_profit_rule(model):
        a = df_params['slope']
        b = df_params['intercept']
        a_d = df_params['deal_slope']
        b_d = df_params['deal_intercept']
        total_profit = 0
        for i in range(n_product):
            # 日常每天利润
            fba_commission = model.prices[i] * 0.15
            manual_fee = model.prices[i] * 0.06
            storage_fee = model.prices[i] * 0.02

            # 每个产品的利润 日常
            product_income = model.prices[i] * (1 - df_params['damage_rate'][i]) - df_params['latest_amazon_delivery_fee_usd'][i] - \
                fba_commission - df_params['vat_local'][i] - df_params['promotion_discount'][i] - \
                df_params['ppc_cost'][i] - manual_fee - storage_fee
            product_profit_rmb = product_income * df_params['exchange_rate'][i] - df_params['purchase_cost'][i] - \
                df_params['shipping_and_tax'][i]

            # 每个产品的利润 deal
            product_income_deal = model.prices[i] * 0.85 * (1 - df_params['damage_rate'][i]) - df_params['latest_amazon_delivery_fee_usd'][i] - \
                fba_commission - df_params['vat_local'][i] - df_params['promotion_discount'][i] - \
                df_params['ppc_cost'][i] - manual_fee - storage_fee
            product_profit_rmb_deal = product_income_deal * df_params['exchange_rate'][i] - df_params['purchase_cost'][i] - \
                df_params['shipping_and_tax'][i]

            # 日常销量
            product_sales = a[i] * model.prices[i] + b[i]

            # deal销量
            product_sales_deal = a_d[i] * model.prices[i] + b_d[i]

            # 总利润
            all_deal_days = model.deal_days['LD'] + model.deal_days['BD']*7
            total_profit += (30 - all_deal_days * model.deal[i]) * product_profit_rmb * product_sales \
                + all_deal_days * model.deal[i] * product_profit_rmb_deal * product_sales_deal \
                - model.deal_days['LD']*150 - \
                model.deal_days['BD']*300  # deal费用

        return total_profit

    model.total_profit = pyo.Objective(
        rule=total_profit_rule, sense=pyo.maximize)

    solver = pyo.SolverFactory('ipopt')
    result = solver.solve(model)

    optimized_prices = [np.round(pyo.value(model.prices[i])).astype(
        int) for i in range(n_product)]
    deal = [np.round(pyo.value(model.deal[i])).astype(int)
            for i in range(n_product)]
    deal_days = [np.round(pyo.value(model.deal_days[i])).astype(int)
                 for i in model.A]

    return optimized_prices, deal, deal_days


def result_process(df_params, optimized_prices, deal, deal_days):
    df_result = df_params.copy()
    df_result['optimized_price'] = optimized_prices
    df_result['expected_sales'] = (
        df_result['slope'] * df_result['optimized_price'] + df_result['intercept']).round(0).astype(int)
    df_result['expected_sales_deal'] = (
        df_result['deal_slope'] * df_result['optimized_price'] + df_result['deal_intercept']).round(0).astype(int)
    df_result['fba_commission_local'] = df_result['optimized_price']*0.15
    df_result['labor_cost_6_percent'] = df_result['optimized_price']*0.06
    df_result['storage_cost_2_percent'] = df_result['optimized_price']*0.02
    # 是否做deal
    df_result['is_deal'] = deal
    df_result['product_income'] = df_result['optimized_price'] * (1 - df_result['damage_rate']) - df_result['latest_amazon_delivery_fee_usd'] - \
        df_result['fba_commission_local'] - df_result['vat_local'] - df_result['promotion_discount'] - \
        df_result['ppc_cost'] - df_result['labor_cost_6_percent'] - \
        df_result['storage_cost_2_percent']
    df_result['product_profit_rmb'] = df_result['product_income'] * df_result['exchange_rate'] - df_result['purchase_cost'] - \
        df_result['shipping_and_tax']
    df_result['product_income_deal'] = df_result['optimized_price'] * 0.85 * (1 - df_result['damage_rate']) - df_result['latest_amazon_delivery_fee_usd'] - \
        df_result['fba_commission_local'] - df_result['vat_local'] - df_result['promotion_discount'] - \
        df_result['ppc_cost'] - df_result['labor_cost_6_percent'] - \
        df_result['storage_cost_2_percent']
    df_result['product_profit_rmb_deal'] = df_result['product_income_deal'] * df_result['exchange_rate'] - df_result['purchase_cost'] - \
        df_result['shipping_and_tax']
    df_result['month_profit_rmb'] = (30 - deal_days[0] - deal_days[1]*7) * df_result['product_profit_rmb'] * df_result['expected_sales'] \
        + deal_days[0] * df_result['product_profit_rmb_deal'] * df_result['expected_sales_deal'] \
        - deal_days[0]*150 - deal_days[1]*300

    df_result = df_result.drop(
        columns=['slope', 'intercept', 'deal_slope', 'deal_intercept'])
    return df_result


@app.post("/optimize", response_model=OptimizeResponse)
def optimize_endpoint(params: List[ProductParams]):
    try:
        df_params = pd.DataFrame([param.model_dump() for param in params])
        df_params = preprocess_file(df_params)
        optimized_prices, deal, deal_days = optimize(df_params)
        df_result = result_process(
            df_params, optimized_prices, deal, deal_days)
        response = OptimizeResponse(
            optimized_price = df_result['optimized_price'].tolist(),
            is_deal = df_result['is_deal'].tolist(),
            month_profit_rmb = df_result['month_profit_rmb'].tolist(),
            ld = deal_days[0],
            bd = deal_days[1]
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=52565)
