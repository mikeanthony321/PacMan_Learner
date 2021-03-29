
BUTTON_STYLE = """
                QPushButton{ 
                text-align: center; 
                border: 1px solid white;
                        }
                QPushButton:disabled{ background-color: yellow; }
                QPushButton:pressed{ background-color: grey; }

                """

QTAB_STYLE = """
                QTabBar::tab {
                background: rgb(0, 0, 0); 
                border: 1px solid white;
                padding: 10px;
                        } 

                QTabBar::tab:selected { 
                background: rgb(40, 40, 40); 
                border: 3px solid rgb(225, 190, 5); 
                }
                """