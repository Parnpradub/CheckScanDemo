import unittest
from ocrengine import views
import cv2

class TestViews(unittest.TestCase):
    image1 = cv2.imread("test1.jpg")
    image1 = cv2.resize(image1, (2000, 800))
    image2 = cv2.imread("test2.jpg")
    image2 = cv2.resize(image2, (2000, 800))
    image3 = cv2.imread("test3.jpg")
    image3 = cv2.resize(image3, (2000, 800))

    def test_get_companyInfo(self):
        #if the result is same as expected value , it is success
        self.assertEqual(views.get_companyInfo(self.image1,0,850,85,140),"CUNNINGHAM ELECTRIC, INC")
        self.assertEqual(views.get_companyInfo(self.image2, 0, 850, 85, 140), "R & H MECHANICAL, LLC")
        self.assertEqual(views.get_companyInfo(self.image3, 0, 850, 85, 140), "BODEC, INC.")

        #check random crop image
        self.assertEqual(views.get_companyInfo(self.image1,0,10,0,10), "")
        self.assertEqual(views.get_companyInfo(self.image2, 0, 10, 5, 10), "")
        self.assertEqual(views.get_companyInfo(self.image3, 0, 10, 0, 15), "")

    def test_get_priceInfo(self):
        #check result
        self.assertEqual(views.get_priceInfo(self.image1,1620,1890,300,450),"15,044.84")
        self.assertEqual(views.get_priceInfo(self.image2, 1620, 1890, 300, 450), "267.54")
        self.assertEqual(views.get_priceInfo(self.image3, 1620, 1890, 300, 450), "355.67")

        #check radom part image
        self.assertEqual(views.get_priceInfo(self.image3, 0, 10, 0, 15), "")
        self.assertEqual(views.get_priceInfo(self.image2, 100, 170, 50, 500), "")
        #result is 48, so error is occured
        self.assertNotEqual(views.get_priceInfo(self.image1, 300, 350, 0, 350), "")

        #all price value should be digit
        self.assertFalse(views.get_priceInfo(self.image1,1620,1890,300,450).isalpha())
        self.assertFalse(views.get_priceInfo(self.image1, 70, 300, 47, 350).isalpha())

    def test_get_bankInfo(self):
        banklist1 = views.get_bankInfo(self.image1, 350, 1500, 725, 800)
        # get routing info from bank Info
        routing1 = banklist1[len(banklist1) - 2]
        # get account info from bankInfo
        account1 = banklist1[len(banklist1) - 1]

        banklist2 = views.get_bankInfo(self.image2, 350, 1500, 725, 800)
        # get routing info from bank Info
        routing2 = banklist2[len(banklist2) - 2]
        # get account info from bankInfo
        account2 = banklist2[len(banklist2) - 1]

        banklist3 = views.get_bankInfo(self.image3, 350, 1500, 725, 800)
        # get routing info from bank Info
        routing3 = banklist3[len(banklist3) - 2]
        # get account info from bankInfo
        account3 = banklist3[len(banklist3) - 1]

        self.assertEqual(account1, "12003471")
        self.assertEqual(account2, "41029")
        self.assertEqual(account3, "77804200")

        self.assertEqual(routing1, "102307119")
        self.assertEqual(routing2, "102003206")
        self.assertEqual(routing3, "324377817")

        #check random part input
        views.get_bankInfo(self.image1, 0, 89, 73, 370)
        views.get_bankInfo(self.image2, 79, 480, 500, 720)

    #integration test
    def test_requested_url(self):
        result1 = views.requested_url("test1.jpg")
        result2 = views.requested_url("test2.jpg")
        result3 = views.requested_url("test3.jpg")

        print(result1)
        print(result2)
        print(result3)


if __name__ == '__main__':
    unittest.main()




