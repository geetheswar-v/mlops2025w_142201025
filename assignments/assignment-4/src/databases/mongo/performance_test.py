import time
import json
from datetime import datetime
from pathlib import Path
from databases.mongo.crud_operations import *

client, db, transactions_collection, transactions_customers_collection, customers_collection = get_mongo_client("local")

# --- Configuration ---
EXISTING_CUSTOMER_ID = 13408
EXISTING_INVOICE_NO = '541883'
ANALYTICS_STOCK_CODE = '85123A'
RESULTS_DIR = Path("doc")
RESULTS_FILE = RESULTS_DIR / "performance_results.json"

NEW_INVOICE_NO = "999999"
NEW_TRANSACTION = {'InvoiceNo': NEW_INVOICE_NO, 
                   'InvoiceDate': datetime.now(), 
                   'CustomerID': EXISTING_CUSTOMER_ID,
                   'Items': [
                       {'StockCode': 'P01', 'Description': 'Test Item', 'Quantity': 1, 'UnitPrice': 10.99}
                       ]
                   }

NEW_SUB_INVOICE = {'InvoiceNo': NEW_INVOICE_NO, 
                   'InvoiceDate': datetime.now(), 
                   'Items': [
                       {'StockCode': 'P01', 'Description': 'Test Item', 'Quantity': 2, 'UnitPrice': 10.99}
                       ]
                   }


# Helper Functions
def run_and_measure(func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    return result, (end_time - start_time) * 1000

def generate_test_transactions(count: int = 100):
    return [{
        'InvoiceNo': str(800000 + i),
        'InvoiceDate': datetime.now(),
        'CustomerID': EXISTING_CUSTOMER_ID + (i % 100),
        'Items': [{
            'StockCode': f'TEST{i:04d}',
            'Description': f'Test Product {i}',
            'Quantity': (i % 10) + 1,
            'UnitPrice': round(5.99 + (i % 20), 2)
        }]
    } for i in range(count)]

def generate_bulk_updates(count: int = 50):
    return [{
        'invoice_no': str(800000 + i),
        'stock_code': f'TEST{i:04d}',
        'new_quantity': (i % 15) + 5
    } for i in range(count)]
    
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

# --- Main Test Execution ---
def run_performance_tests():
    print("Starting MongoDB Performance Tests...")

    # Define comparative tests
    comparative_tests = {
        # Create Operation
        "Create Invoice": (
            (create_new_transaction, [NEW_TRANSACTION]),
            (add_invoice_to_customer, [EXISTING_CUSTOMER_ID, NEW_SUB_INVOICE])
        ),
        
        # Read Operations
        "Find by ID": (
            (find_invoice_by_id_transaction, [EXISTING_INVOICE_NO]), 
            (find_invoice_by_id_customer, [EXISTING_INVOICE_NO])
        ),
        "Single Invoice": (
            (find_invoice_by_id_transaction, [EXISTING_INVOICE_NO]), 
            (find_invoice_by_id_customer, [EXISTING_INVOICE_NO])
        ),
        "Customer Invoices": (
            (find_invoices_by_customer_id_transaction, [EXISTING_CUSTOMER_ID]),
            (find_invoices_by_customer_id_customer, [EXISTING_CUSTOMER_ID])
        ),
        "Customer Invoices by Month/Year": (
            (get_customer_invoices_by_month_year_transaction, [EXISTING_CUSTOMER_ID, 5, 2011]),
            (get_customer_invoices_by_month_year_customer, [EXISTING_CUSTOMER_ID, 5, 2011])
        ),
        "Product Sales": (
            (get_total_sales_for_product_transaction, [ANALYTICS_STOCK_CODE]),
            (get_total_sales_for_product_customer, [ANALYTICS_STOCK_CODE])
        ),
        "Top Customers": (
            (get_top_customers_by_revenue_transaction, [10]),
            (get_top_customers_by_revenue_customer, [10])
        ),
        "Sales by Country": (
            (get_sales_by_country_transaction, []),
            (get_sales_by_country_customer, [])
        ),
        "Product Search": (
            (search_products_by_description_transaction, ["CHRISTMAS", 20]),
            (search_products_by_description_customer, ["CHRISTMAS", 20])
        ),
        
        # Update Operation
        "Update Item Quantity": (
            (update_item_quantity_transaction, [EXISTING_INVOICE_NO, ANALYTICS_STOCK_CODE, 50]),
            (update_item_quantity_customer, [EXISTING_CUSTOMER_ID, EXISTING_INVOICE_NO, ANALYTICS_STOCK_CODE, 50])
        ),
        
        # Delete Operation
        "Delete Invoice": (
            (delete_invoice_transaction, [NEW_INVOICE_NO]),
            (delete_invoice_customer, [NEW_INVOICE_NO])
        ),
    }

    for name, (tx_test, cust_test) in comparative_tests.items():
        _, t_tx = run_and_measure(tx_test[0], *tx_test[1])
        _, t_cust = run_and_measure(cust_test[0], *cust_test[1])
        compare_times(name, t_tx, t_cust)

    # Bulk Operations
    transactions = generate_test_transactions(100)
    _, t_bulk_insert = run_and_measure(bulk_insert_transactions, transactions)
    print(f"Bulk Insert (100 records): {t_bulk_insert:.2f}ms")

    updates = generate_bulk_updates(50)
    _, t_bulk_update = run_and_measure(bulk_update_quantities_transaction, updates)
    print(f"Bulk Update (50 records): {t_bulk_update:.2f}ms")

    # Cleanup
    test_invoice_nos = [t['InvoiceNo'] for t in transactions]
    _, t_bulk_delete = run_and_measure(bulk_delete_transactions, test_invoice_nos)
    print(f"Bulk Delete (100 records): {t_bulk_delete:.2f}ms")
    
    delete_invoice_transaction(NEW_INVOICE_NO)
    delete_invoice_customer(NEW_INVOICE_NO)

def main():
    run_performance_tests()
    print("\nTest Completed.")

if __name__ == "__main__":
    main()