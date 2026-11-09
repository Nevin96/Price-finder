from project_mcp.mcp_server.recommendation_engine import predict_buying_decision

res = predict_buying_decision(
    current_price=64999.0,
    lowest_price_so_far=49999.0,
    category="smartphone",
    store="Amazon India",
    avg_price=59999.0,
)
print(res)