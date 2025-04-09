"""
Author: Varun Balaji
SID: 3036383355
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from BlackScholes import BlackScholes
from GeometricAsian import geometric_asian_option_price
from GeometricBasket import geometric_basket_option_price
import plotly.graph_objects as go



# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "Option Pricer"


# App layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.H1("Option Pricer", className="text-center my-4"), width=12)
    ]),
    
    # Tabs
    dbc.Tabs([
        # Tab 1: Black-Scholes Pricer
        dbc.Tab([
            dbc.Card([
            dbc.CardBody([
            html.H4("Black-Scholes Option Pricing", className="card-title"),
            dbc.Row([
            dbc.Col([
                dbc.Label("Spot Price (S)"),
                dbc.Input(id='bs-spot', type='number', value=100),
            ], md=3),
            dbc.Col([
                dbc.Label("Volatility (σ)"),
                dbc.Input(id='bs-vol', type='number', value=0.2),
            ], md=3),
            dbc.Col([
                dbc.Label("Risk-Free Rate (r)"),
                dbc.Input(id='bs-rate', type='number', value=0.05),
            ], md=3),
            dbc.Col([
                dbc.Label("Repo Rate (q)"),
                dbc.Input(id='bs-repo', type='number', value=0.01),
            ], md=3),
            ], className="mb-3"),
            dbc.Row([
            dbc.Col([
                dbc.Label("Time to Maturity (T)"),
                dbc.Input(id='bs-time', type='number', value=1),
            ], md=3),
            dbc.Col([
                dbc.Label("Strike Price (K)"),
                dbc.Input(id='bs-strike', type='number', value=100),
            ], md=3),
            dbc.Col([
                dbc.Label("Option Type"),
                dcc.Dropdown(
                id='bs-type',
                options=[{'label': 'Call', 'value': 'call'}, {'label': 'Put', 'value': 'put'}],
                value='call'
                ),
            ], md=3),
            dbc.Col([
                dbc.Button("Calculate", id='bs-calculate', color="primary", className="mt-4"),
            ], md=3),
            ]),
            html.Hr(),
            dbc.Row([
            dbc.Col([
                html.H5("Option Price:"),
                html.Div(id='bs-price', className="h4 text-success")
            ], className="text-center")
            ]),
            html.Hr(style={"borderTop": "2px solid #dee2e6"}),  # Add a horizontal divider
            html.Div([
            html.P(
                "The Black-Scholes model is a mathematical model used for pricing European-style options. "
                "It assumes that the price of the underlying asset follows a geometric Brownian motion with constant volatility and drift.",
                className="mt-3"
            ),
            html.P("The formula for the price of a call option is given by:", className="mt-3"),
            html.Div([
                html.Span("C = S * N(d₁) - K * e⁻ʳᵀ * N(d₂)", style={"fontWeight": "bold", "fontSize": "1.2em"})
            ], className="text-center mt-2"),
            html.P("For a put option, the formula is:", className="mt-3"),
            html.Div([
                html.Span("P = K * e⁻ʳᵀ * N(-d₂) - S * N(-d₁)", style={"fontWeight": "bold", "fontSize": "1.2em"})
            ], className="text-center mt-2"),
            html.P("where d₁ and d₂ are intermediate calculations.", className="mt-3")
            ], style={"marginTop": "40px"})
            ])
            ], className="mt-4")
        ], label="Black-Scholes"),
        
        # Tab 2: Implied Volatility Calculator
        dbc.Tab([
            dbc.Card([
            dbc.CardBody([
            html.H4("Implied Volatility Calculator", className="card-title"),
            dbc.Row([
                dbc.Col([
                dbc.Label("Spot Price (S)"),
                dbc.Input(id='iv-spot', type='number', value=100),
                ], md=3),
                dbc.Col([
                dbc.Label("Risk-Free Rate (r)"),
                dbc.Input(id='iv-rate', type='number', value=0.05),
                ], md=3),
                dbc.Col([
                dbc.Label("Repo Rate (q)"),
                dbc.Input(id='iv-repo', type='number', value=0.01),
                ], md=3),
                dbc.Col([
                dbc.Label("Time to Maturity (T)"),
                dbc.Input(id='iv-time', type='number', value=1),
                ], md=3),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                dbc.Label("Strike Price (K)"),
                dbc.Input(id='iv-strike', type='number', value=100),
                ], md=3),
                dbc.Col([
                dbc.Label("Option Premium"),
                dbc.Input(id='iv-premium', type='number', value=10),
                ], md=3),
                dbc.Col([
                dbc.Label("Option Type"),
                dcc.Dropdown(
                    id='iv-type',
                    options=[{'label': 'Call', 'value': 'call'}, {'label': 'Put', 'value': 'put'}],
                    value='call',
                    style={"width": "100%"}  # Ensure dropdown matches the size of other inputs
                ),
                ], md=3),
                dbc.Col([
                dbc.Button("Calculate", id='iv-calculate', color="primary", className="mt-4"),
                ], md=3),
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                html.H5("Implied Volatility:"),
                html.Div(id='iv-result', className="h4 text-success")
                ], className="text-center")
            ]),
            html.Hr(style={"borderTop": "2px solid #dee2e6"}),  # Add a horizontal divider
            html.Div([
                html.P(
                "Implied volatility is the volatility value that, when input into an option pricing model, "
                "yields the observed market price of the option. It is often calculated using numerical methods.",
                className="mt-3"
                ),
                html.P(
                "In this calculator, we use the Newton-Raphson method to iteratively solve for the implied volatility. "
                "The formula for updating the volatility estimate is:",
                className="mt-3"
                ),
                html.Div([
                html.Span("σₙ₊₁ = σₙ - f(σₙ) / f'(σₙ)", style={"fontWeight": "bold", "fontSize": "1.2em"})
                ], className="text-center mt-2"),
                html.P(
                "Here, f(σ) is the difference between the market price and the model price of the option, "
                "and f'(σ) is the derivative of f(σ) with respect to σ.",
                className="mt-3"
                ),
                html.Div([
                html.Span("f(σ) = C(σ) - Actual Value", style={"fontWeight": "bold", "fontSize": "1.2em"}),
                ], className="text-center mt-2"),
                html.P(
                "In this context, f'(σ) is also known as vega, which measures the sensitivity of the option price "
                "to changes in volatility.",
                className="mt-3"
                ),
                html.P(
                "This method converges quickly for well-behaved functions, making it suitable for implied volatility calculations.",
                className="mt-3"
                )
            ], style={"marginTop": "40px"})
            ])
            ], className="mt-4")
        ], label="Implied Volatility"),
        
        
        # Tab 3: Closed Form Geometric Option
        dbc.Tab([
            dbc.Card([
            dbc.CardBody([
                html.H4("Closed Form Geometric Asian Option", className="card-title"),
                dbc.Row([
                dbc.Col([
                    dbc.Label("Spot Price (S)"),
                    dbc.Input(id='geo-spot', type='number', value=100),
                ], md=3),
                dbc.Col([
                    dbc.Label("Volatility (σ)"),
                    dbc.Input(id='geo-vol', type='number', value=0.2),
                ], md=3),
                dbc.Col([
                    dbc.Label("Risk-Free Rate (r)"),
                    dbc.Input(id='geo-rate', type='number', value=0.05),
                ], md=3),
                dbc.Col([
                    dbc.Label("Time to Maturity (T)"),
                    dbc.Input(id='geo-time', type='number', value=1),
                ], md=3),
                ], className="mb-3"),
                dbc.Row([
                dbc.Col([
                    dbc.Label("Strike Price (K)"),
                    dbc.Input(id='geo-strike', type='number', value=100),
                ], md=3),
                dbc.Col([
                    dbc.Label("Number of Observations (n)"),
                    dbc.Input(id='geo-obs', type='number', value=12),
                ], md=3),
                dbc.Col([
                    dbc.Label("Option Type"),
                    dcc.Dropdown(
                    id='geo-type',
                    options=[{'label': 'Call', 'value': 'call'}, {'label': 'Put', 'value': 'put'}],
                    value='call'
                    ),
                ], md=3),
                dbc.Col([
                    dbc.Button("Calculate", id='geo-calculate', color="primary", className="mt-4"),
                ], md=3),
                ]),
                html.Hr(),
                dbc.Row([
                dbc.Col([
                    html.H5("Option Price:"),
                    html.Div(id='geo-price', className="h4 text-success")
                ], className="text-center")
                ]),
                html.Hr(style={"borderTop": "2px solid #dee2e6"}),  # Add a horizontal divider
                html.Div([
                html.P(
                    "The closed-form solution for geometric Asian options is derived under the assumption "
                    "that the average of the underlying asset prices follows a lognormal distribution.",
                    className="mt-3"
                ),
                html.P("The formula for the price of a geometric Asian call option is:", className="mt-3"),
                html.Div([
                    html.Span("C = e⁻ʳᵀ [S * eᵃ * N(d₁) - K * N(d₂)]", style={"fontWeight": "bold", "fontSize": "1.2em"})
                ], className="text-center mt-2"),
                html.P("For a put option, the formula is:", className="mt-3"),
                html.Div([
                    html.Span("P = e⁻ʳᵀ [K * N(-d₂) - S * eᵃ * N(-d₁)]", style={"fontWeight": "bold", "fontSize": "1.2em"})
                ], className="text-center mt-2"),
                html.P(
                    "Here, d₁ and d₂ are intermediate calculations, and a is an adjustment factor "
                    "accounting for the averaging process.",
                    className="mt-3"
                )
                ], style={"marginTop": "40px"})
            ])
            ], className="mt-4")
        ], label="Geometric Asian"),
       
        
        # Tab 4: Closed Form Basket Option
        dbc.Tab([
            dbc.Card([
            dbc.CardBody([
                html.H4("Closed Form Basket Option", className="card-title"),
                dbc.Row([
                dbc.Col([
                    dbc.Label("Spot Price 1 (S1)"),
                    dbc.Input(id='basket-s1', type='number', value=100),
                ], md=3),
                dbc.Col([
                    dbc.Label("Spot Price 2 (S2)"),
                    dbc.Input(id='basket-s2', type='number', value=100),
                ], md=3),
                dbc.Col([
                    dbc.Label("Volatility 1 (σ1)"),
                    dbc.Input(id='basket-vol1', type='number', value=0.2),
                ], md=3),
                dbc.Col([
                    dbc.Label("Volatility 2 (σ2)"),
                    dbc.Input(id='basket-vol2', type='number', value=0.2),
                ], md=3),
                ], className="mb-3"),
                dbc.Row([
                dbc.Col([
                    dbc.Label("Risk-Free Rate (r)"),
                    dbc.Input(id='basket-rate', type='number', value=0.05),
                ], md=3),
                dbc.Col([
                    dbc.Label("Time to Maturity (T)"),
                    dbc.Input(id='basket-time', type='number', value=1),
                ], md=3),
                dbc.Col([
                    dbc.Label("Strike Price (K)"),
                    dbc.Input(id='basket-strike', type='number', value=100),
                ], md=3),
                dbc.Col([
                    dbc.Label("Correlation (ρ)"),
                    dbc.Input(id='basket-correlation', type='number', value=0.5, step=0.01, min=-1, max=1),
                ], md=3),
                ]),
                dbc.Row([
                dbc.Col([
                    dbc.Label("Option Type"),
                    dcc.Dropdown(
                    id='basket-type',
                    options=[{'label': 'Call', 'value': 'call'}, {'label': 'Put', 'value': 'put'}],
                    value='call'
                    ),
                ], md=3),
                dbc.Col([
                    dbc.Button("Calculate", id='basket-calculate', color="primary", className="mt-4"),
                ], md=3),
                ]),
                html.Hr(),
                dbc.Row([
                dbc.Col([
                    html.H5("Option Price:"),
                    html.Div(id='basket-price', className="h4 text-success")
                ], className="text-center")
                ]),
                html.Hr(style={"borderTop": "2px solid #dee2e6"}),  # Add a horizontal divider
                html.Div([
                html.P(
                    "The closed-form solution for basket options assumes that the weighted sum of the underlying "
                    "assets follows a lognormal distribution. This approach simplifies the pricing of basket options.",
                    className="mt-3"
                ),
                html.P("The formula for the price of a basket call option is:", className="mt-3"),
                html.Div([
                    html.Span("C = e⁻ʳᵀ [B * N(d₁) - K * N(d₂)]", style={"fontWeight": "bold", "fontSize": "1.2em"})
                ], className="text-center mt-2"),
                html.P("For a put option, the formula is:", className="mt-3"),
                html.Div([
                    html.Span("P = e⁻ʳᵀ [K * N(-d₂) - B * N(-d₁)]", style={"fontWeight": "bold", "fontSize": "1.2em"})
                ], className="text-center mt-2"),
                html.P(
                    "Here, B is the weighted average of the spot prices, and d₁ and d₂ are intermediate calculations "
                    "that depend on the volatilities, correlation, and weights of the assets.",
                    className="mt-3"
                )
                ], style={"marginTop": "40px"})
            ])
            ], className="mt-4")
        ], label="Basket Option"),
        
        # Tab 5: Monte Carlo Arithmetic Asian
        dbc.Tab([
            dbc.Card([
            dbc.CardBody([
                html.H4("Monte Carlo Arithmetic Asian Option", className="card-title"),
                dbc.Row([
                dbc.Col([
                    dbc.Label("Spot Price (S)"),
                    dbc.Input(id='mc-aa-spot', type='number', value=100),
                ], md=3),
                dbc.Col([
                    dbc.Label("Volatility (σ)"),
                    dbc.Input(id='mc-aa-vol', type='number', value=0.2),
                ], md=3),
                dbc.Col([
                    dbc.Label("Risk-Free Rate (r)"),
                    dbc.Input(id='mc-aa-rate', type='number', value=0.05),
                ], md=3),
                dbc.Col([
                    dbc.Label("Time to Maturity (T)"),
                    dbc.Input(id='mc-aa-time', type='number', value=1),
                ], md=3),
                ], className="mb-3"),
                dbc.Row([
                dbc.Col([
                    dbc.Label("Strike Price (K)"),
                    dbc.Input(id='mc-aa-strike', type='number', value=100),
                ], md=3),
                dbc.Col([
                    dbc.Label("Number of Observations (n)"),
                    dbc.Input(id='mc-aa-obs', type='number', value=12),
                ], md=3),
                dbc.Col([
                    dbc.Label("Number of Paths"),
                    dbc.Input(id='mc-aa-paths', type='number', value=10000),
                ], md=3),
                dbc.Col([
                    dbc.Label("Option Type"),
                    dcc.Dropdown(
                    id='mc-aa-type',
                    options=[{'label': 'Call', 'value': 'call'}, {'label': 'Put', 'value': 'put'}],
                    value='call'
                    ),
                ], md=3),
                ]),
                dbc.Row([
                dbc.Col([
                    dbc.Label("Use Control Variate?"),
                    dcc.Dropdown(
                    id='mc-aa-cv',
                    options=[{'label': 'Yes', 'value': 'yes'}, {'label': 'No', 'value': 'no'}],
                    value='no'
                    ),
                ], md=3),
                dbc.Col([
                    dbc.Button("Calculate", id='mc-aa-calculate', color="primary", className="mt-4"),
                ], md=3),
                ]),
                html.Hr(),
                dbc.Row([
                dbc.Col([
                    html.H5("Option Price:"),
                    html.Div(id='mc-aa-price', className="h4 text-success mb-3"),
                    html.H5("95% Confidence Interval:"),
                    html.Div(id='mc-aa-ci', className="h4 text-info")
                ], className="text-center")
                ]),
                html.Hr(style={"borderTop": "2px solid #dee2e6"}),  # Add a horizontal divider
                html.Div([
                html.P(
                    "The Monte Carlo method is a numerical technique used to estimate the price of options "
                    "by simulating the paths of the underlying asset's price. It is particularly useful for "
                    "pricing options with complex payoffs, such as arithmetic Asian options.",
                    className="mt-3"
                ),
                html.P(
                    "For an arithmetic Asian call option, the payoff is based on the average price of the "
                    "underlying asset over a specified period. The formula for the payoff is:",
                    className="mt-3"
                ),
                html.Div([
                    html.Span("Payoff = max(0, A - K)", style={"fontWeight": "bold", "fontSize": "1.2em"}),
                    html.P("where A is the arithmetic average of the underlying prices, and K is the strike price.")
                ], className="text-center mt-2"),
                html.P(
                    "The Monte Carlo method involves generating multiple random paths for the underlying asset's price, "
                    "calculating the payoff for each path, and then averaging the discounted payoffs to estimate the option price.",
                    className="mt-3"
                ),
                html.P(
                    "To improve accuracy, the control variate technique can be used. This method reduces variance by "
                    "leveraging the known price of a related option, such as a geometric Asian option.",
                    className="mt-3"
                )
                ], style={"marginTop": "40px"})
            ])
            ], className="mt-4")
        ], label="MC Arithmetic Asian"),
        
        # Tab 6: Monte Carlo Arithmetic Basket
        dbc.Tab([
            dbc.Card([
            dbc.CardBody([
                html.H4("Monte Carlo Arithmetic Basket Option", className="card-title"),
                dbc.Row([
                dbc.Col([
                    dbc.Label("Spot Price 1 (S1)"),
                    dbc.Input(id='mc-ab-s1', type='number', value=100),
                ], md=3),
                dbc.Col([
                    dbc.Label("Spot Price 2 (S2)"),
                    dbc.Input(id='mc-ab-s2', type='number', value=100),
                ], md=3),
                dbc.Col([
                    dbc.Label("Volatility 1 (σ1)"),
                    dbc.Input(id='mc-ab-vol1', type='number', value=0.2),
                ], md=3),
                dbc.Col([
                    dbc.Label("Volatility 2 (σ2)"),
                    dbc.Input(id='mc-ab-vol2', type='number', value=0.2),
                ], md=3),
                ], className="mb-3"),
                dbc.Row([
                dbc.Col([
                    dbc.Label("Risk-Free Rate (r)"),
                    dbc.Input(id='mc-ab-rate', type='number', value=0.05),
                ], md=3),
                dbc.Col([
                    dbc.Label("Time to Maturity (T)"),
                    dbc.Input(id='mc-ab-time', type='number', value=1),
                ], md=3),
                dbc.Col([
                    dbc.Label("Strike Price (K)"),
                    dbc.Input(id='mc-ab-strike', type='number', value=100),
                ], md=3),
                dbc.Col([
                    dbc.Label("Correlation (ρ)"),
                    dbc.Input(id='mc-ab-correlation', type='number', value=0.5, step=0.01, min=-1, max=1),
                ], md=3),
                ]),
                dbc.Row([
                dbc.Col([
                    dbc.Label("Number of Paths"),
                    dbc.Input(id='mc-ab-paths', type='number', value=10000),
                ], md=3),
                dbc.Col([
                    dbc.Label("Option Type"),
                    dcc.Dropdown(
                    id='mc-ab-type',
                    options=[{'label': 'Call', 'value': 'call'}, {'label': 'Put', 'value': 'put'}],
                    value='call'
                    ),
                ], md=3),
                dbc.Col([
                    dbc.Label("Use Control Variate?"),
                    dcc.Dropdown(
                    id='mc-ab-cv',
                    options=[{'label': 'Yes', 'value': 'yes'}, {'label': 'No', 'value': 'no'}],
                    value='no'
                    ),
                ], md=3),
                dbc.Col([
                    dbc.Button("Calculate", id='mc-ab-calculate', color="primary", className="mt-4"),
                ], md=3),
                ]),
                html.Hr(),
                dbc.Row([
                dbc.Col([
                    html.H5("Option Price:"),
                    html.Div(id='mc-ab-price', className="h4 text-success mb-3"),
                    html.H5("95% Confidence Interval:"),
                    html.Div(id='mc-ab-ci', className="h4 text-info")
                ], className="text-center")
                ]),
                html.Hr(style={"borderTop": "2px solid #dee2e6"}),  # Add a horizontal divider
                html.Div([
                html.P(
                    "The Monte Carlo method is a powerful numerical technique used to estimate the price of "
                    "complex options, such as arithmetic basket options. It involves simulating multiple random "
                    "paths for the underlying assets' prices and calculating the average payoff.",
                    className="mt-3"
                ),
                html.P(
                    "For an arithmetic basket call option, the payoff is based on the arithmetic average of the "
                    "underlying assets' prices. The formula for the payoff is:",
                    className="mt-3"
                ),
                html.Div([
                    html.Span("Payoff = max(0, A - K)", style={"fontWeight": "bold", "fontSize": "1.2em"}),
                    html.P("where A is the arithmetic average of the basket prices, and K is the strike price.")
                ], className="text-center mt-2"),
                html.P(
                    "The Monte Carlo method calculates the option price by averaging the discounted payoffs "
                    "over all simulated paths. To improve accuracy, the control variate technique can be used, "
                    "leveraging the known price of a related option, such as a geometric basket option.",
                    className="mt-3"
                ),
                html.P(
                    "This method is particularly useful for options with complex payoffs, where closed-form "
                    "solutions are not available.",
                    className="mt-3"
                )
                ], style={"marginTop": "40px"})
            ])
            ], className="mt-4")
        ], label="MC Arithmetic Basket"),
        
        # Tab 7: Quasi Monte Carlo KIKO
        dbc.Tab([
            dbc.Card([
            dbc.CardBody([
                html.H4("Quasi Monte Carlo KIKO Option", className="card-title"),
                dbc.Row([
                dbc.Col([
                    dbc.Label("Spot Price (S)"),
                    dbc.Input(id='kiko-spot', type='number', value=100),
                ], md=3),
                dbc.Col([
                    dbc.Label("Volatility (σ)"),
                    dbc.Input(id='kiko-vol', type='number', value=0.2),
                ], md=3),
                dbc.Col([
                    dbc.Label("Risk-Free Rate (r)"),
                    dbc.Input(id='kiko-rate', type='number', value=0.05),
                ], md=3),
                dbc.Col([
                    dbc.Label("Time to Maturity (T)"),
                    dbc.Input(id='kiko-time', type='number', value=1),
                ], md=3),
                ], className="mb-3"),
                dbc.Row([
                dbc.Col([
                    dbc.Label("Strike Price (K)"),
                    dbc.Input(id='kiko-strike', type='number', value=100),
                ], md=3),
                dbc.Col([
                    dbc.Label("Lower Barrier (L)"),
                    dbc.Input(id='kiko-lower', type='number', value=90),
                ], md=3),
                dbc.Col([
                    dbc.Label("Upper Barrier (U)"),
                    dbc.Input(id='kiko-upper', type='number', value=110),
                ], md=3),
                dbc.Col([
                    dbc.Label("Number of Observations (n)"),
                    dbc.Input(id='kiko-obs', type='number', value=252),
                ], md=3),
                ]),
                dbc.Row([
                dbc.Col([
                    dbc.Label("Cash Rebate (R)"),
                    dbc.Input(id='kiko-rebate', type='number', value=0),
                ], md=3),
                dbc.Col([
                    dbc.Button("Calculate", id='kiko-calculate', color="primary", className="mt-4"),
                ], md=3),
                ]),
                html.Hr(),
                dbc.Row([
                dbc.Col([
                    html.H5("Option Price:"),
                    html.Div(id='kiko-price', className="h4 text-success mb-3"),
                    html.H5("Option Delta:"),
                    html.Div(id='kiko-delta', className="h4 text-info")
                ], className="text-center")
                ]),
                html.Hr(style={"borderTop": "2px solid #dee2e6"}),  # Add a horizontal divider
                html.Div([
                html.P(
                    "The Quasi Monte Carlo (QMC) method is an advanced numerical technique used to estimate "
                    "the price of options by leveraging low-discrepancy sequences. Unlike traditional Monte Carlo methods, "
                    "which rely on pseudo-random numbers, QMC uses deterministic sequences to achieve faster convergence.",
                    className="mt-3"
                ),
                html.P(
                    "For a Knock-In Knock-Out (KIKO) option, the payoff depends on whether the underlying asset's price "
                    "remains within specified barriers during the option's lifetime. The formula for the payoff is:",
                    className="mt-3"
                ),
                html.Div([
                    html.Span("Payoff = max(0, S - K) if L < S < U, else Rebate", style={"fontWeight": "bold", "fontSize": "1.2em"})
                ], className="text-center mt-2"),
                html.P(
                    "Here, S is the spot price, K is the strike price, L is the lower barrier, U is the upper barrier, "
                    "and Rebate is the cash rebate paid if the barriers are breached.",
                    className="mt-3"
                ),
                html.P(
                    "The QMC method improves accuracy by reducing the variance of the simulation. It achieves this by "
                    "using low-discrepancy sequences, such as Sobol or Halton sequences, which are more evenly distributed "
                    "than random numbers. This makes QMC particularly effective for pricing exotic options like KIKO options.",
                    className="mt-3"
                ),
                html.P(
                    "By combining the QMC method with advanced variance reduction techniques, the pricing of KIKO options "
                    "becomes both efficient and accurate, even for complex scenarios with multiple barriers and observations.",
                    className="mt-3"
                )
                ], style={"marginTop": "40px"})
            ])
            ], className="mt-4")
        ], label="QMC KIKO"),
        
        # Tab 8: Binomial Tree American
        dbc.Tab([
            dbc.Card([
            dbc.CardBody([
                html.H4("Binomial Tree for American Option", className="card-title"),
                dbc.Row([
                dbc.Col([
                    dbc.Label("Spot Price (S)"),
                    dbc.Input(id='bt-spot', type='number', value=100),
                ], md=3),
                dbc.Col([
                    dbc.Label("Volatility (σ)"),
                    dbc.Input(id='bt-vol', type='number', value=0.2),
                ], md=3),
                dbc.Col([
                    dbc.Label("Risk-Free Rate (r)"),
                    dbc.Input(id='bt-rate', type='number', value=0.05),
                ], md=3),
                dbc.Col([
                    dbc.Label("Time to Maturity (T)"),
                    dbc.Input(id='bt-time', type='number', value=1),
                ], md=3),
                ], className="mb-3"),
                dbc.Row([
                dbc.Col([
                    dbc.Label("Strike Price (K)"),
                    dbc.Input(id='bt-strike', type='number', value=100),
                ], md=3),
                dbc.Col([
                    dbc.Label("Number of Steps (N)"),
                    dbc.Input(id='bt-steps', type='number', value=100),
                ], md=3),
                dbc.Col([
                    dbc.Label("Option Type"),
                    dcc.Dropdown(
                    id='bt-type',
                    options=[{'label': 'Call', 'value': 'call'}, {'label': 'Put', 'value': 'put'}],
                    value='put'
                    ),
                ], md=3),
                dbc.Col([
                    dbc.Button("Calculate", id='bt-calculate', color="primary", className="mt-4"),
                ], md=3),
                ]),
                html.Hr(),
                dbc.Row([
                dbc.Col([
                    html.H5("Option Price:"),
                    html.Div(id='bt-price', className="h4 text-success")
                ], className="text-center")
                ]),
                html.Hr(style={"borderTop": "2px solid #dee2e6"}),  # Add a horizontal divider
                html.Div([
                html.P(
                    "The Binomial Tree model is a numerical method used to price options by discretizing the "
                    "time to maturity into smaller intervals. At each step, the price of the underlying asset "
                    "can either move up or down by a certain factor, creating a tree-like structure.",
                    className="mt-3"
                ),
                html.P(
                    "For an American option, the holder has the right to exercise the option at any point before "
                    "maturity. The Binomial Tree model accounts for this by checking the option's intrinsic value "
                    "at each node and comparing it to the continuation value.",
                    className="mt-3"
                ),
                html.P("The key formulas used in the Binomial Tree model are:", className="mt-3"),
                html.Div([
                    html.Span("u = e^(σ√Δt), d = 1/u", style={"fontWeight": "bold", "fontSize": "1.2em"}),
                    html.P("where u and d are the up and down factors, σ is the volatility, and Δt is the time step.")
                ], className="text-center mt-2"),
                html.Div([
                    html.Span("p = (e^(rΔt) - d) / (u - d)", style={"fontWeight": "bold", "fontSize": "1.2em"}),
                    html.P("where p is the risk-neutral probability of an upward movement.")
                ], className="text-center mt-2"),
                html.P(
                    "The option price is calculated by working backward through the tree, starting from the terminal "
                    "nodes and applying the risk-neutral valuation principle at each step.",
                    className="mt-3"
                )
                ], style={"marginTop": "40px"})
            ])
            ], className="mt-4")
        ], label="Binomial Tree"),
        
        # Tab 9: News
        dbc.Tab([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Stock News", className="card-title"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Stock Symbol"),
                            dbc.Input(id='news-symbol', type='text', value='AAPL'),
                        ], md=6),
                        dbc.Col([
                            dbc.Button("Get News", id='news-get', color="primary", className="mt-4"),
                        ], md=6),
                    ]),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.Div(id='news-results')
                        ])
                    ])
                ])
            ], className="mt-4")
        ], label="News"),
        
        # Tab 10: Contact Us
        dbc.Tab([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Contact Us", className="card-title"),
                    html.P("For any questions or support, please reach out to our team:"),
                    html.Ul([
                        html.Li("Email: support@optionpricer.com"),
                        html.Li("Phone: +1 (555) 123-4567"),
                        html.Li("Address: 123 Finance Street, New York, NY 10001")
                    ]),
                    html.Hr(),
                    html.H5("Send us a message:"),
                    dbc.Form([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Your Name"),
                                dbc.Input(type="text", placeholder="Enter your name"),
                            ], md=6),
                            dbc.Col([
                                dbc.Label("Your Email"),
                                dbc.Input(type="email", placeholder="Enter your email"),
                            ], md=6),
                        ], className="mb-3"),
                        dbc.Label("Message"),
                        dbc.Textarea(placeholder="Your message here...", rows=5),
                        dbc.Button("Submit", color="primary", className="mt-3"),
                    ])
                ])
            ], className="mt-4")
        ], label="Contact Us")
    ])
], fluid=True, style={'backgroundColor': '#f8f9fa'})

# Callbacks for each tab
# Note: These are placeholder callbacks that you'll need to connect to your backend

# Black-Scholes Callback
@callback(
    Output('bs-price', 'children'),
    Input('bs-calculate', 'n_clicks'),
    [State('bs-spot', 'value'),
     State('bs-vol', 'value'),
     State('bs-rate', 'value'),
     State('bs-repo', 'value'),
     State('bs-time', 'value'),
     State('bs-strike', 'value'),
     State('bs-type', 'value')]
)
def calculate_bs(n_clicks, spot, vol, rate, repo, time, strike, option_type):
    if n_clicks is None:
        return ""
    
    try:
        bs_model = BlackScholes(
            S=spot,
            K=strike,
            T=time,
            t=0,
            r=rate,
            q=repo
        )

        if option_type == 'call':
            price = bs_model.call(sigma=vol)
        elif option_type == 'put':
            price =  bs_model.put(sigma=vol)
    except Exception as e:
        return f"Error: {str(e)}"

    return f"${price:.2f}"

# Implied Volatility Callback
@callback(
    Output('iv-result', 'children'),
    Input('iv-calculate', 'n_clicks'),
    [State('iv-spot', 'value'),
     State('iv-rate', 'value'),
     State('iv-repo', 'value'),
     State('iv-time', 'value'),
     State('iv-strike', 'value'),
     State('iv-premium', 'value'),
     State('iv-type', 'value')]
)
def calculate_iv(n_clicks, spot, rate, repo, time, strike, premium, option_type):
    if n_clicks is None:
        return ""
    
    try:
        bs_model = BlackScholes(
            S=spot,
            K=strike,
            T=time,
            t=0,
            r=rate,
            q=repo
        )

        if option_type == 'call':
            iv = bs_model.implied_volatility(Ctrue=premium, OptionType='C')
        elif option_type == 'put':
            iv = bs_model.implied_volatility(Ctrue=premium, OptionType='P')
    except Exception as e:
        return f"Error: {str(e)}"
    
    
    return f"{iv*100:.2f}%"

# Geometric Asian Callback
@callback(
    Output('geo-price', 'children'),
    Input('geo-calculate', 'n_clicks'),
    [State('geo-spot', 'value'),
     State('geo-vol', 'value'),
     State('geo-rate', 'value'),
     State('geo-time', 'value'),
     State('geo-strike', 'value'),
     State('geo-obs', 'value'),
     State('geo-type', 'value')]
)
def calculate_geo(n_clicks, spot, vol, rate, time, strike, obs, option_type):
    if n_clicks is None:
        return ""
    
    try:
        price = geometric_asian_option_price(
            S0=spot,
            sigma=vol,
            r=rate,
            T=time,
            K=strike,
            n=obs,
            option_type=option_type
        )
    except Exception as e:
        return f"Error: {str(e)}"
    
    return f"${price:.2f}"

# Basket Option Callback
@callback(
    Output('basket-price', 'children'),
    Input('basket-calculate', 'n_clicks'),
    [State('basket-s1', 'value'),
     State('basket-s2', 'value'),
     State('basket-vol1', 'value'),
     State('basket-vol2', 'value'),
     State('basket-rate', 'value'),
     State('basket-time', 'value'),
     State('basket-strike', 'value'),
     State('basket-correlation', 'value'),
     State('basket-type', 'value')]
)
def calculate_basket(n_clicks, s1, s2, vol1, vol2, rate, time, strike, correlation, option_type):
    if n_clicks is None:
        return ""
    
    try:
        price = geometric_basket_option_price(
            S1_0=s1,
            S2_0=s2,
            sigma1=vol1,
            sigma2=vol2,
            r=rate,
            T=time,
            K=strike,
            rho=correlation,
            type=option_type
        )
    except Exception as e:
        return f"Error: {str(e)}"
    
    return f"${price:.2f}"

# Monte Carlo Arithmetic Asian Callback
@callback(
    [Output('mc-aa-price', 'children'),
     Output('mc-aa-ci', 'children')],
    Input('mc-aa-calculate', 'n_clicks'),
    [State('mc-aa-spot', 'value'),
     State('mc-aa-vol', 'value'),
     State('mc-aa-rate', 'value'),
     State('mc-aa-time', 'value'),
     State('mc-aa-strike', 'value'),
     State('mc-aa-obs', 'value'),
     State('mc-aa-paths', 'value'),
     State('mc-aa-type', 'value'),
     State('mc-aa-cv', 'value')]
)
def calculate_mc_aa(n_clicks, spot, vol, rate, time, strike, obs, paths, option_type, cv):
    if n_clicks is None:
        return "", ""
    
    # Here you would call your backend API
    # For now, just return placeholders
    price = 9.85  # Replace with actual API call
    ci_low = 9.72
    ci_high = 9.98
    
    return f"${price:.2f}", f"[${ci_low:.2f}, ${ci_high:.2f}]"

# Monte Carlo Arithmetic Basket Callback
@callback(
    [Output('mc-ab-price', 'children'),
     Output('mc-ab-ci', 'children')],
    Input('mc-ab-calculate', 'n_clicks'),
    [State('mc-ab-s1', 'value'),
     State('mc-ab-s2', 'value'),
     State('mc-ab-vol1', 'value'),
     State('mc-ab-vol2', 'value'),
     State('mc-ab-rate', 'value'),
     State('mc-ab-time', 'value'),
     State('mc-ab-strike', 'value'),
     State('mc-ab-correlation', 'value'),
     State('mc-ab-paths', 'value'),
     State('mc-ab-type', 'value'),
     State('mc-ab-cv', 'value')]
)
def calculate_mc_ab(n_clicks, s1, s2, vol1, vol2, rate, time, strike, correlation, paths, option_type, cv):
    if n_clicks is None:
        return "", ""
    
    # Here you would call your backend API
    # For now, just return placeholders
    price = 13.20  # Replace with actual API call
    ci_low = 13.05
    ci_high = 13.35
    
    return f"${price:.2f}", f"[${ci_low:.2f}, ${ci_high:.2f}]"

# KIKO Option Callback
@callback(
    [Output('kiko-price', 'children'),
     Output('kiko-delta', 'children')],
    Input('kiko-calculate', 'n_clicks'),
    [State('kiko-spot', 'value'),
     State('kiko-vol', 'value'),
     State('kiko-rate', 'value'),
     State('kiko-time', 'value'),
     State('kiko-strike', 'value'),
     State('kiko-lower', 'value'),
     State('kiko-upper', 'value'),
     State('kiko-obs', 'value'),
     State('kiko-rebate', 'value')]
)
def calculate_kiko(n_clicks, spot, vol, rate, time, strike, lower, upper, obs, rebate):
    if n_clicks is None:
        return "", ""
    
    # Here you would call your backend API
    # For now, just return placeholders
    price = 5.60  # Replace with actual API call
    delta = 0.45
    
    return f"${price:.2f}", f"{delta:.4f}"

# Binomial Tree Callback
@callback(
    Output('bt-price', 'children'),
    Input('bt-calculate', 'n_clicks'),
    [State('bt-spot', 'value'),
     State('bt-vol', 'value'),
     State('bt-rate', 'value'),
     State('bt-time', 'value'),
     State('bt-strike', 'value'),
     State('bt-steps', 'value'),
     State('bt-type', 'value')]
)
def calculate_bt(n_clicks, spot, vol, rate, time, strike, steps, option_type):
    if n_clicks is None:
        return ""
    
    # Here you would call your backend API
    # For now, just return a placeholder
    price = 7.25  # Replace with actual API call
    
    return f"${price:.2f}"

# News Callback
@callback(
    Output('news-results', 'children'),
    Input('news-get', 'n_clicks'),
    State('news-symbol', 'value')
)
def get_news(n_clicks, symbol):
    if n_clicks is None:
        return ""
    
    # Here you would call a news API
    # For now, just return placeholder news items
    news_items = [
        {"title": f"{symbol} reports strong Q3 earnings", "date": "2023-11-15", "source": "Financial Times"},
        {"title": f"Analysts raise price target for {symbol}", "date": "2023-11-10", "source": "Bloomberg"},
        {"title": f"{symbol} announces new product line", "date": "2023-11-05", "source": "Wall Street Journal"}
    ]
    
    return dbc.ListGroup([
        dbc.ListGroupItem([
            html.H5(item["title"]),
            html.Small(f"{item['source']} - {item['date']}")
        ]) for item in news_items
    ])

if __name__ == '__main__':
    app.run(debug=True)