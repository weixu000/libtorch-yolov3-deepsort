#include <wx/wxprec.h>

#ifndef WX_PRECOMP

#include <wx/wx.h>

#endif

#include <wx/progdlg.h>
#include <opencv2/opencv.hpp>

#include "thumbnailctrl.h"
#include "TargetRepo.h"
#include "util.h"
#include "wxplayer.h"

class MyApp : public wxApp {
public:
    bool OnInit() override;
};

wxIMPLEMENT_APP(MyApp);

class MyFrame : public wxFrame {
public:
    MyFrame(const wxString &dir);

private:
    void InitMenu();

    wxThumbnailCtrl *InitThumbnails();

    wxPlayer *player;

    TargetRepo repo;

    int hovered = -1;

    enum {
        ID_Hello = 1,
        ID_Timer,
        ID_List,
    };
};


bool MyApp::OnInit() {
    auto dialog = wxDirDialog(nullptr,
                              "Select result directory",
                              "",
                              wxDD_DIR_MUST_EXIST);
    if (dialog.ShowModal() == wxID_OK) {
        auto frame = new MyFrame(dialog.GetPath());
        frame->Show(true);
        return true;
    } else {
        return false;
    }
}

MyFrame::MyFrame(const wxString &dir)
        : wxFrame(nullptr, wxID_ANY, "YOLO+DeepSORT+wxWidgets"),
          repo(dir.ToStdString()) {
    InitMenu();

    player = new wxPlayer(this, wxID_ANY, repo.video_path(),
                          [this](cv::Mat &mat, int display_frame) {
                              for (auto &[id, t]:repo.get()) {
                                  if (t.trajectories.count(display_frame)) {
                                      auto color =
                                              hovered == -1 ? color_map(id) : hovered == id ? cv::Scalar(0, 0, 255)
                                                                                            : cv::Scalar(0, 0, 0);
                                      draw_trajectories(mat, t.trajectories, display_frame, color);
                                      draw_bbox(mat, t.trajectories.at(display_frame),
                                                std::to_string(id), color);
                                  }
                              }
                          });

    auto sizer = new wxBoxSizer(wxHORIZONTAL);
    sizer->Add(player, 3, wxEXPAND | wxALL);
    sizer->Add(InitThumbnails(), 1, wxEXPAND | wxALL);
    SetSizerAndFit(sizer);
}

void MyFrame::InitMenu() {
    auto menuFile = new wxMenu;
    menuFile->Append(ID_Hello, "&Hello...\tCtrl-H",
                     "Help string shown in status bar for this menu item");
    menuFile->AppendSeparator();
    menuFile->Append(wxID_EXIT);

    auto menuHelp = new wxMenu;
    menuHelp->Append(wxID_ABOUT);

    auto menuBar = new wxMenuBar;
    menuBar->Append(menuFile, "&File");
    menuBar->Append(menuHelp, "&Help");
    SetMenuBar(menuBar);

    CreateStatusBar();
    SetStatusText("Welcome to my demo!");

    Bind(wxEVT_MENU,
         [this](wxCommandEvent &) {
             wxLogMessage("Hello world!");
         }, ID_Hello);
    Bind(wxEVT_MENU,
         [this](wxCommandEvent &) {
             wxMessageBox("This is a wxWidgets Hello World example",
                          "About Hello World", wxOK | wxICON_INFORMATION);
         }, wxID_ABOUT);
    Bind(wxEVT_MENU,
         [this](wxCommandEvent &) {
             Close(true);
         }, wxID_EXIT);
}

wxThumbnailCtrl *MyFrame::InitThumbnails() {
    auto dialog = wxProgressDialog("Loading results", wxEmptyString);
    repo.load([&dialog](int value) { dialog.Update(value / 2, "Loading targets..."); });
    auto thumbnails = new wxThumbnailCtrl(this, ID_List);
    thumbnails->SetThumbnailImageSize(wxSize(50, 50));
    int i_target = 0;
    for (auto &[id, t]:repo.get()) {
        for (auto &[s_t, s]:t.snapshots) {
            cv::cvtColor(s, s, cv::COLOR_BGR2RGB);
            cv::resize(s, s, cv::Size(50, 50));
        }
        auto item = new wxThumbnailItem(wxString::Format("%d", id));
        item->SetBitmap(cvMat2wxImage(t.snapshots.begin()->second));
        thumbnails->Append(item);
        dialog.Update(50 + 50 * ++i_target / repo.get().size(), "Loading resources...");
    }

    static std::map<int, cv::Mat>::const_iterator it{};
    auto timer = new wxTimer(thumbnails, ID_Timer);
    thumbnails->Bind(wxEVT_TIMER,
                     [this, thumbnails](wxTimerEvent &) {
                         if (thumbnails->GetMouseHoverItem() != wxNOT_FOUND) {
                             auto &item = *thumbnails->GetItem(thumbnails->GetMouseHoverItem());
                             auto &snapshots = repo.get().at(wxAtoi(item.GetLabel())).snapshots;
                             item.SetBitmap(cvMat2wxImage(it->second));
                             item.Refresh(thumbnails, thumbnails->GetMouseHoverItem());
                             if (++it == snapshots.end()) {
                                 it = snapshots.begin();
                             }
                         }
                     }, ID_Timer);
    timer->Start(1000 * 5 / player->GetFPS());
    thumbnails->Bind(wxEVT_COMMAND_THUMBNAIL_ITEM_HOVER_CHANGED,
                     [this, thumbnails](wxThumbnailEvent &event) {
                         if (event.GetIndex() != wxNOT_FOUND) {
                             auto &item = *thumbnails->GetItem(event.GetIndex());
                             item.SetBitmap(cvMat2wxImage(
                                     repo.get().at(wxAtoi(item.GetLabel())).snapshots.begin()->second));
                         }

                         if (thumbnails->GetMouseHoverItem() != wxNOT_FOUND) {
                             auto &item = *thumbnails->GetItem(thumbnails->GetMouseHoverItem());
                             it = repo.get().at(wxAtoi(item.GetLabel())).snapshots.begin();
                         }
                     }, ID_List);

    thumbnails->Bind(wxEVT_COMMAND_THUMBNAIL_ITEM_SELECTED,
                     [this, thumbnails](wxThumbnailEvent &event) {
                         auto id = wxAtoi(thumbnails->GetItem(event.GetIndex())->GetLabel());
                         player->Seek(repo.get().at(id).trajectories.begin()->first);
                     }, ID_List);

    thumbnails->Bind(wxEVT_COMMAND_THUMBNAIL_ITEM_HOVER_CHANGED,
                     [this, thumbnails](wxThumbnailEvent &event) {
                         if (thumbnails->GetMouseHoverItem() != wxNOT_FOUND) {
                             hovered = wxAtoi(thumbnails->GetItem(thumbnails->GetMouseHoverItem())->GetLabel());
                             player->Refresh();
                         } else {
                             hovered = -1;
                             player->Refresh();
                         }
                         Refresh();
                         event.Skip();
                     }, ID_List);

    return thumbnails;
}
