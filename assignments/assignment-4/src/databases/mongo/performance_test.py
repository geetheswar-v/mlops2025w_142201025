import time
from datetime import datetime
from databases.mongo.crud_operations import *

# Test Data
EXISTING_CUSTOMER_ID = 13408
EXISTING_INVOICE_NO = '541883'
EXISTING_STOCK_CODE = '22866'
ANALYTICS_STOCK_CODE = '85123A'

NEW_INVOICE_NO = "999999"

NEW_TRANSACTION_DOC = {
    'InvoiceNo': NEW_INVOICE_NO,
    'InvoiceDate': datetime.now(),
    'CustomerID': EXISTING_CUSTOMER_ID,
    'Country': 'United Kingdom',
    'Items': [{'StockCode': 'TEST01', 'Description': 'TEST ITEM', 'Quantity': 5, 'UnitPrice': 9.99}]
}

NEW_INVOICE_SUB_DOC = {
    'InvoiceNo': NEW_INVOICE_NO,
    'InvoiceDate': datetime.now(),
    'Items': [{'StockCode': 'TEST02', 'Description': 'TEST ITEM 2', 'Quantity': 1, 'UnitPrice': 19.99}]
}

# --- Helper Functions ---
def run_and_measure(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, (end - start) * 1000

def compare_times(name, t_tx, t_cust):
    print(f"\n{name}")
    print(f"  Transaction-Centric: {t_tx:.2f} ms")
    print(f"  Customer-Centric:    {t_cust:.2f} ms")
    if t_tx < t_cust:
        print(f"  -> Transaction faster ({t_cust/t_tx:.2f}x)")
    elif t_cust < t_tx:
        print(f"  -> Customer faster ({t_tx/t_cust:.2f}x)")
    else:
        print("  -> Equal performance")

def main():
    print("MongoDB Performance Comparison\n")

    # CREATE
    _, t_tx = run_and_measure(create_new_transaction, NEW_TRANSACTION_DOC)
    _, t_cust = run_and_measure(add_invoice_to_customer, EXISTING_CUSTOMER_ID, NEW_INVOICE_SUB_DOC)
    compare_times("Create Invoice", t_tx, t_cust)

    # READ
    _, t_tx = run_and_measure(find_invoice_by_id_transaction, EXISTING_INVOICE_NO)
    _, t_cust = run_and_measure(find_invoice_by_id_customer, EXISTING_INVOICE_NO)
    compare_times("Find Invoice by ID", t_tx, t_cust)

    _, t_tx = run_and_measure(find_invoices_by_customer_id_transaction, EXISTING_CUSTOMER_ID)
    _, t_cust = run_and_measure(find_invoices_by_customer_id_customer, EXISTING_CUSTOMER_ID)
    compare_times("Get All Invoices by Customer", t_tx, t_cust)

    # READ by month/year
    year, month = 2011, 12
    _, t_tx = run_and_measure(get_customer_invoices_by_month_year_transaction, EXISTING_CUSTOMER_ID, month, year)
    _, t_cust = run_and_measure(get_customer_invoices_by_month_year_customer, EXISTING_CUSTOMER_ID, month, year)
    compare_times(f"Get Invoices for {month}/{year}", t_tx, t_cust)

    # UPDATE
    _, t_tx = run_and_measure(update_item_quantity_transaction, EXISTING_INVOICE_NO, EXISTING_STOCK_CODE, 100)
    _, t_cust = run_and_measure(update_item_quantity_customer, EXISTING_CUSTOMER_ID, EXISTING_INVOICE_NO, EXISTING_STOCK_CODE, 100)
    compare_times("Update Item Quantity", t_tx, t_cust)

    # AGGREGATE
    _, t_tx = run_and_measure(get_total_sales_for_product_transaction, ANALYTICS_STOCK_CODE)
    _, t_cust = run_and_measure(get_total_sales_for_product_customer, ANALYTICS_STOCK_CODE)
    compare_times(f"Total Sales for Product {ANALYTICS_STOCK_CODE}", t_tx, t_cust)

    # DELETE
    _, t_tx = run_and_measure(delete_invoice_transaction, NEW_INVOICE_NO)
    _, t_cust = run_and_measure(delete_invoice_customer, NEW_INVOICE_NO)
    compare_times("Delete Invoice", t_tx, t_cust)

    print("\nTest Completed.")

if __name__ == "__main__":
    main()
