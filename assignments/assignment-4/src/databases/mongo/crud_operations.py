from datetime import datetime
from pymongo import UpdateOne
from databases.mongo import get_mongo_client, handle_mongo_errors

client, db, transactions_collection, transactions_customers_collection, customers_collection = get_mongo_client("local")

#%% CREATE OPERATIONS

@handle_mongo_errors
def create_new_transaction(transaction_data: dict):
    if transactions_collection is None:
        print("Error: Collections not initialized. Call initialize_collections() first.")
        return None
    return transactions_collection.insert_one(transaction_data)

@handle_mongo_errors
def create_new_customer(customer_data: dict):
    if transactions_customers_collection is None:
        print("Error: Collections not initialized.")
        return None
    return transactions_customers_collection.insert_one(customer_data)

@handle_mongo_errors
def add_invoice_to_customer(customer_id: int, invoice_data: dict):
    if customers_collection is None:
        print("Error: Collections not initialized. Call initialize_collections() first.")
        return None
    return customers_collection.update_one(
        {'CustomerID': customer_id},
        {'$push': {'Invoices': invoice_data}}
    )


#%% READ OPERATIONS
#%% Helper: Get Customer Info from transactions_customers
@handle_mongo_errors
def get_customer_info_transaction(customer_id: int):
    return transactions_customers_collection.find_one({'CustomerID': customer_id})

#%% Case 1: Find Invoice by InvoiceNo
@handle_mongo_errors
def find_invoice_by_id_transaction(invoice_no: str, include_customer_info: bool = False):
    if not include_customer_info:
        return transactions_collection.find_one({'InvoiceNo': invoice_no})
    
    # Use aggregation with $lookup to join with transactions_customers
    pipeline = [
        {'$match': {'InvoiceNo': invoice_no}},
        {'$lookup': {
            'from': 'transactions_customers',
            'localField': 'CustomerID',
            'foreignField': 'CustomerID',
            'as': 'customer_info'
        }},
        {'$unwind': {'path': '$customer_info', 'preserveNullAndEmptyArrays': True}}
    ]
    result = list(transactions_collection.aggregate(pipeline))
    return result[0] if result else None

@handle_mongo_errors
def find_invoice_by_id_customer(invoice_no: str):
    return customers_collection.find_one({'Invoices.InvoiceNo': invoice_no})

#%% Case 2: Find Invoices made by Customer by CustomerID
@handle_mongo_errors
def find_invoices_by_customer_id_transaction(customer_id: int):
    return list(transactions_collection.find({'CustomerID': customer_id}))

@handle_mongo_errors
def find_invoices_by_customer_id_customer(customer_id: int):
    return customers_collection.find_one({'CustomerID': customer_id})

#%% Case 3: Get Customer Invoices by Month and Year
@handle_mongo_errors
def get_customer_invoices_by_month_year_transaction(customer_id: int, month: int, year: int):
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)
    
    return list(transactions_collection.find({
        'CustomerID': customer_id,
        'InvoiceDate': {'$gte': start_date, '$lt': end_date}
    }))
    
@handle_mongo_errors
def get_customer_invoices_by_month_year_customer(customer_id: int, month: int, year: int):
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)
    
    customer = customers_collection.find_one({'CustomerID': customer_id})
    if not customer or 'Invoices' not in customer:
        return []
    
    filtered_invoices = [
        inv for inv in customer['Invoices']
        if start_date <= inv['InvoiceDate'] < end_date
    ]
    return filtered_invoices

#%% Case 4: Total Sales for a Product
@handle_mongo_errors
def get_total_sales_for_product_transaction(stock_code: str):
    pipeline = [
        {'$unwind': '$Items'},
        {'$match': {'Items.StockCode': stock_code}},
        {'$group': {
            '_id': '$Items.StockCode',
            'totalQuantity': {'$sum': '$Items.Quantity'},
            'totalRevenue': {'$sum': {'$multiply': ['$Items.Quantity', '$Items.UnitPrice']}}
        }}
    ]
    return list(transactions_collection.aggregate(pipeline))

@handle_mongo_errors
def get_total_sales_for_product_customer(stock_code: str):
    pipeline = [
        {'$unwind': '$Invoices'},
        {'$unwind': '$Invoices.Items'},
        {'$match': {'Invoices.Items.StockCode': stock_code}},
        {'$group': {
            '_id': '$Invoices.Items.StockCode',
            'totalQuantity': {'$sum': '$Invoices.Items.Quantity'},
            'totalRevenue': {'$sum': {'$multiply': ['$Invoices.Items.Quantity', '$Invoices.Items.UnitPrice']}}
        }}
    ]
    return list(customers_collection.aggregate(pipeline))

#%% UPDATE OPERATIONS
@handle_mongo_errors
def update_item_quantity_transaction(invoice_no: str, stock_code: str, new_quantity: int):
    return transactions_collection.update_one(
        {'InvoiceNo': invoice_no, 'Items.StockCode': stock_code},
        {'$set': {'Items.$.Quantity': new_quantity}}
    )

@handle_mongo_errors
def update_item_quantity_customer(customer_id: int, invoice_no: str, stock_code: str, new_quantity: int):
    return customers_collection.update_one(
        {'CustomerID': customer_id},
        {'$set': {'Invoices.$[inv].Items.$[itm].Quantity': new_quantity}},
        array_filters=[
            {'inv.InvoiceNo': invoice_no},
            {'itm.StockCode': stock_code}
        ]
    )

#%% DELETE OPERATIONS
@handle_mongo_errors
def delete_invoice_transaction(invoice_no: str):
    return transactions_collection.delete_one({'InvoiceNo': invoice_no})

@handle_mongo_errors
def delete_invoice_customer(invoice_no: str):
    return customers_collection.update_one(
        {'Invoices.InvoiceNo': invoice_no},
        {'$pull': {'Invoices': {'InvoiceNo': invoice_no}}}
    )

#%% BULK OPERATIONS
@handle_mongo_errors
def bulk_insert_transactions(transactions: list[dict]):
    if transactions_collection is None:
        print("Error: Collections not initialized. Call initialize_collections() first.")
        return None
    return transactions_collection.insert_many(transactions, ordered=False)

@handle_mongo_errors
def bulk_update_quantities_transaction(updates: list[dict]):
    bulk_ops = []
    for update in updates:
        bulk_ops.append(UpdateOne(
            {'InvoiceNo': update['invoice_no'], 'Items.StockCode': update['stock_code']},
            {'$set': {'Items.$.Quantity': update['new_quantity']}}
        ))
    return transactions_collection.bulk_write(bulk_ops, ordered=False)

@handle_mongo_errors
def bulk_delete_transactions(invoice_nos: list[str]):
    return transactions_collection.delete_many({'InvoiceNo': {'$in': invoice_nos}})

#%% ADVANCED QUERY OPERATIONS
@handle_mongo_errors
def get_top_customers_by_revenue_transaction(limit: int = 10):
    pipeline = [
        {'$unwind': '$Items'},
        {'$group': {
            '_id': '$CustomerID',
            'totalRevenue': {'$sum': {'$multiply': ['$Items.Quantity', '$Items.UnitPrice']}},
            'totalOrders': {'$sum': 1}
        }},
        {'$sort': {'totalRevenue': -1}},
        {'$limit': limit}
    ]
    return list(transactions_collection.aggregate(pipeline))

@handle_mongo_errors
def get_top_customers_by_revenue_customer(limit: int = 10):
    pipeline = [
        {'$unwind': '$Invoices'},
        {'$unwind': '$Invoices.Items'},
        {'$group': {
            '_id': '$CustomerID',
            'totalRevenue': {'$sum': {'$multiply': ['$Invoices.Items.Quantity', '$Invoices.Items.UnitPrice']}},
            'totalOrders': {'$sum': 1}
        }},
        {'$sort': {'totalRevenue': -1}},
        {'$limit': limit}
    ]
    return list(customers_collection.aggregate(pipeline))

@handle_mongo_errors
def get_sales_by_country_transaction():
    """
    Transaction-centric approach: Use $lookup to join with transactions_customers.
    This demonstrates proper normalization where Country is stored separately.
    """
    pipeline = [
        # Join with transactions_customers to get Country
        {'$lookup': {
            'from': 'transactions_customers',
            'localField': 'CustomerID',
            'foreignField': 'CustomerID',
            'as': 'customer_info'
        }},
        {'$unwind': '$customer_info'},
        {'$unwind': '$Items'},
        {'$group': {
            '_id': '$customer_info.Country',
            'totalRevenue': {'$sum': {'$multiply': ['$Items.Quantity', '$Items.UnitPrice']}},
            'totalQuantity': {'$sum': '$Items.Quantity'},
            'uniqueCustomers': {'$addToSet': '$CustomerID'}
        }},
        {'$project': {
            '_id': 1,
            'totalRevenue': 1,
            'totalQuantity': 1,
            'uniqueCustomers': {'$size': '$uniqueCustomers'}
        }},
        {'$sort': {'totalRevenue': -1}}
    ]
    return list(transactions_collection.aggregate(pipeline))

@handle_mongo_errors
def get_sales_by_country_customer():
    pipeline = [
        {'$unwind': '$Invoices'},
        {'$unwind': '$Invoices.Items'},
        {'$group': {
            '_id': '$Country',
            'totalRevenue': {'$sum': {'$multiply': ['$Invoices.Items.Quantity', '$Invoices.Items.UnitPrice']}},
            'totalQuantity': {'$sum': '$Invoices.Items.Quantity'},
            'uniqueCustomers': {'$addToSet': '$CustomerID'}
        }},
        {'$project': {
            '_id': 1,
            'totalRevenue': 1,
            'totalQuantity': 1,
            'uniqueCustomers': {'$size': '$uniqueCustomers'}
        }},
        {'$sort': {'totalRevenue': -1}}
    ]
    return list(customers_collection.aggregate(pipeline))

@handle_mongo_errors
def search_products_by_description_transaction(search_term: str, limit: int = 20):
    return list(transactions_collection.find(
        {'Items.Description': {'$regex': search_term, '$options': 'i'}},
        {'InvoiceNo': 1, 'Items': 1}
    ).limit(limit))

@handle_mongo_errors
def search_products_by_description_customer(search_term: str, limit: int = 20):
    return list(customers_collection.find(
        {'Invoices.Items.Description': {'$regex': search_term, '$options': 'i'}},
        {'CustomerID': 1, 'Invoices.Items': 1}
    ).limit(limit))