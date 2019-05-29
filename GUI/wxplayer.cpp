#include "wxplayer.h"
#include "util.h"

wxPlayer::wxPlayer(wxWindow *parent, wxWindowID id,
                   const wxString &file,
                   const std::function<void(cv::Mat &, int)> &post)
        : wxWindow(parent, id), post(post) {
    if (!capture.open(file.ToStdString())) {
        throw std::runtime_error("Cannot open video!");
    }

    auto video_size = wxSize(capture.get(cv::CAP_PROP_FRAME_WIDTH), capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    mat = cv::Mat::zeros(video_size.GetWidth(), video_size.GetHeight(), CV_8UC3);
    bitmap = new wxGenericStaticBitmap(this, wxID_ANY, wxNullBitmap);
    bitmap->Bind(wxEVT_SIZE, [this](wxSizeEvent &) { RescaleToBitmap(); });
    bitmap->Bind(wxEVT_ERASE_BACKGROUND, [](wxEraseEvent& event) {});
    bitmap->SetMinClientSize(video_size / 3);

    timer = new wxTimer(this, ID_Timer);
    Bind(wxEVT_TIMER,
         [this](wxTimerEvent &) { LoadNext(); },
         ID_Timer);

    auto play = new wxButton(this, ID_Start_Pause, "Start/Pause", wxDefaultPosition, wxDefaultSize, wxBU_EXACTFIT);
    auto stop = new wxButton(this, ID_Reload, "Reload", wxDefaultPosition, wxDefaultSize, wxBU_EXACTFIT);
    progress = new wxSlider(this, ID_Progress, 0, 0, capture.get(cv::CAP_PROP_FRAME_COUNT) - 1);
    auto bar = new wxBoxSizer(wxHORIZONTAL);
    bar->Add(play, 0, wxEXPAND | wxALL);
    bar->Add(stop, 0, wxEXPAND | wxALL);
    bar->Add(progress, 1, wxEXPAND | wxALL);

    Bind(wxEVT_BUTTON, [this](wxCommandEvent &) {
        if (timer->IsRunning()) {
            timer->Stop();
        } else {
            timer->Start(1000 / GetFPS());
        }
    }, ID_Start_Pause);
    Bind(wxEVT_BUTTON, [this](wxCommandEvent &) {
        Seek(0);
    }, ID_Reload);
    Bind(wxEVT_SLIDER, [this](wxCommandEvent &) {
        if (GetFrame() != progress->GetValue()) {
            Seek(progress->GetValue());
        }
    }, ID_Progress);

    auto sizer = new wxBoxSizer(wxVERTICAL);
    sizer->Add(bitmap, 1, wxEXPAND | wxALL);
    sizer->Add(bar, 0, wxEXPAND | wxALL);

    SetSizerAndFit(sizer);

    LoadNext();
}

void wxPlayer::LoadNext() {
    if (capture.read(mat)) {
        RescaleToBitmap();
        progress->SetValue(GetFrame());
    }
}

void wxPlayer::Seek(int frame) {
    capture.set(cv::CAP_PROP_POS_FRAMES, frame);
    LoadNext();
}

void wxPlayer::RescaleToBitmap() {
    cv::Mat resized;
    auto size = bitmap->GetClientSize();
    cv::resize(mat, resized, {size.GetWidth(), size.GetHeight()});
    post(resized, GetFrame());
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    bitmap->SetBitmap(wxBitmap(cvMat2wxImage(resized)));
}
