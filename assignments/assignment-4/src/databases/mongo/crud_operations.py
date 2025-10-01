from datetime import datetime
from databases.mongo import transactions_collection, customers_collection, handle_mongo_errors

#%% CREATE OPERATIONS

@handle_mongo_errors
def create_new_transaction(transaction_data: dict):
    return transactions_collection.insert_one(transaction_data)

@handle_mongo_errors
def add_invoice_to_customer(customer_id: int, invoice_data: dict):
    return customers_collection.update_one(
        {'CustomerID': customer_id},
        {'$push': {'Invoices': invoice_data}}
    )


#%% READ OPERATIONS
#%% Case 1: Find Invoice by InvoiceNo
@handle_mongo_errors
def find_invoice_by_id_transaction(invoice_no: str):
    return transactions_collection.find_one({'InvoiceNo': invoice_no})

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