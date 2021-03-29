
BUTTON_STYLE = """
                QPushButton{ 
                text-align: center; 
                border: 1px outset white;
                font: 12px "Consolas";
                        }
                QPushButton:disabled{ background-color: lightgrey; }
                QPushButton:pressed{ background-color: grey; }

                """
TITLE_STYLE = """
                    font: 16px "Consolas";
                    color: white;
                    """

TEXT_STYLE = """  
                    font: 14px "Consolas";
                    color: white;
                    background: black;
                    """
TABLE_STYLE = """ 
                    QTableView{
                    font: 14px "Consolas";
                    color: white;
                    background: rgb(40, 40, 40); 
                    padding: 5px 2px 5px 2px
                    }
                    
                    QHeaderView::section{ 
                    font: 14px "Consolas";
                    color: white;
                    background-color: black;
                    }

                    """
WIDGET_STYLE = """
                    BorderWidget{
                    border:1px inset white;
                    background: rgb(40, 40, 40); 
                    }
                    
                    QLabel{
                    font: italic 12px "Consolas";
                    color: white;
                    background: rgb(40, 40, 40); 
                    padding: 5px 0px 5px 0px
                    }
                    """
QTAB_STYLE = """
                QTabBar::tab {
                background: rgb(0, 0, 0); 
                border: 1px solid white;
                padding: 10px;
                font: 12px "Consolas";
                color: white;
                min-width: 40ex;
                        } 

                QTabBar::tab:selected { 
                background: rgb(40, 40, 40); 
                border: 3px solid rgb(225, 190, 5); 
                font: 12px "Consolas";
                color: white;
                min-width: 40ex;
                }
                """
