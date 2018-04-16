/* Generated file, do not edit */

#ifndef CXXTEST_RUNNING
#define CXXTEST_RUNNING
#endif

#define _CXXTEST_HAVE_STD
#include <cxxtest/TestListener.h>
#include <cxxtest/TestTracker.h>
#include <cxxtest/TestRunner.h>
#include <cxxtest/RealDescriptions.h>
#include <cxxtest/TestMain.h>
#include <cxxtest/ErrorPrinter.h>

int main( int argc, char *argv[] ) {
 int status;
    CxxTest::ErrorPrinter tmp;
    CxxTest::RealWorldDescription::_worldName = "cxxtest";
    status = CxxTest::Main< CxxTest::ErrorPrinter >( tmp, argc, argv );
    return status;
}
bool suite_MyTestSuite1_init = false;
#include "TestSuite1.h"

static MyTestSuite1 suite_MyTestSuite1;

static CxxTest::List Tests_MyTestSuite1 = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_MyTestSuite1( "TestSuite1.h", 22, "MyTestSuite1", suite_MyTestSuite1, Tests_MyTestSuite1 );

static class TestDescription_suite_MyTestSuite1_testReadBasket : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite1_testReadBasket() : CxxTest::RealTestDescription( Tests_MyTestSuite1, suiteDescription_MyTestSuite1, 26, "testReadBasket" ) {}
 void runTest() { suite_MyTestSuite1.testReadBasket(); }
} testDescription_suite_MyTestSuite1_testReadBasket;

static class TestDescription_suite_MyTestSuite1_testReadViews : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite1_testReadViews() : CxxTest::RealTestDescription( Tests_MyTestSuite1, suiteDescription_MyTestSuite1, 39, "testReadViews" ) {}
 void runTest() { suite_MyTestSuite1.testReadViews(); }
} testDescription_suite_MyTestSuite1_testReadViews;

static class TestDescription_suite_MyTestSuite1_testReadItemFeatures : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite1_testReadItemFeatures() : CxxTest::RealTestDescription( Tests_MyTestSuite1, suiteDescription_MyTestSuite1, 55, "testReadItemFeatures" ) {}
 void runTest() { suite_MyTestSuite1.testReadItemFeatures(); }
} testDescription_suite_MyTestSuite1_testReadItemFeatures;

static class TestDescription_suite_MyTestSuite1_testRun : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite1_testRun() : CxxTest::RealTestDescription( Tests_MyTestSuite1, suiteDescription_MyTestSuite1, 77, "testRun" ) {}
 void runTest() { suite_MyTestSuite1.testRun(); }
} testDescription_suite_MyTestSuite1_testRun;

static class TestDescription_suite_MyTestSuite1_testRunRank : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite1_testRunRank() : CxxTest::RealTestDescription( Tests_MyTestSuite1, suiteDescription_MyTestSuite1, 90, "testRunRank" ) {}
 void runTest() { suite_MyTestSuite1.testRunRank(); }
} testDescription_suite_MyTestSuite1_testRunRank;

#include <cxxtest/Root.cpp>
const char* CxxTest::RealWorldDescription::_worldName = "cxxtest";
